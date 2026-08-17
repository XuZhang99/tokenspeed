# Inference Flow

本文按当前代码说明请求从 API 到 GPU forward 再回到客户端的路径。一个重要变化是：
detokenization 现在内联在主进程 `AsyncLLM`，不再启动独立 Detokenizer 子进程。

## 总览

```text
HTTP / Engine API / SMG
        │
        ▼
AsyncLLM（主进程）
  ├─ tokenize / validate
  ├─ per-request async stream state
  ├─ inline detokenize / output dispatch
  └─ EngineCoreClient（ZMQ）
        │
        ▼
EventLoop worker（每个计算 rank）
  ├─ RequestHandler
  ├─ C++ tokenspeed_scheduler.Scheduler
  ├─ CacheArena + target/draft CachePool
  ├─ ModelExecutor
  │   ├─ ModelRunner
  │   ├─ eager / decode CUDA graph / prefill graph
  │   ├─ speculative drafter
  │   ├─ logits / grammar / sampling
  │   └─ runtime state
  ├─ L2 cache executor
  └─ PD/EPD executor（按配置）
        │
        ▼
token/logprob/finish output → AsyncLLM → client stream
```

## 1. 服务入口与进程拓扑

### 1.1 CLI / SMG

`python -m tokenspeed` 的 serve 路径位于 `python/tokenspeed/cli/`。生产 HTTP/gateway
通常由 `serve_smg.py` 启动 SMG 与 TokenSpeed headless scheduler；它负责参数默认值、
端口、子进程监督、健康探测和控制服务。

Headless 路径调用 `launch_scheduler_headless()`，强制：

```text
zmq_msgpack = True
skip_tokenizer_init = True
```

SMG 拥有 tokenizer/detokenizer，scheduler worker 通过 msgpack ZMQ 与 gateway 连接。

### 1.2 Python `Engine`

[`runtime/entrypoints/engine.py`](../python/tokenspeed/runtime/entrypoints/engine.py)
的 `Engine` 是 Python API 入口。`_launch_subprocesses()`：

- 非 attention-DP：为每个本节点计算 rank 启动 `run_event_loop` 子进程；
- attention-DP：启动 `DataParallelController`，由它管理 rank worker；
- node rank 0 在主进程创建 `AsyncLLM`；其他节点只等待 worker ready；
- `AsyncLLM` 内联 detokenizer，不存在单独 detokenizer process。

worker 用 pipe 回报 model ready、max request length、max single-request tokens 等信息，
frontend 在全部 ready 后才接收流量。

## 2. `AsyncLLM` 前端

[`runtime/engine/async_llm.py`](../python/tokenspeed/runtime/engine/async_llm.py)
负责：

- 创建/持有 tokenizer（`skip_tokenizer_init` 时为空）；
- 将 text、input ids、sampling/logprob/grammar 参数转为 tokenized request；
- 维护 `rid_to_state` 与每请求 async output queue；
- 通过 `EngineCoreClient` 向 scheduler worker 发送请求/控制消息；
- 接收 token ids、logprob、hidden state、finish reason；
- 增量 detokenize、停止字符串处理并向调用方 yield；
- abort、flush cache、pause/resume、profile、weight update 等控制面。

`Engine.generate()` 是同步 facade，内部由 `LLM` 的后台 asyncio loop 桥接；
`Engine.async_generate()` 直接返回 async iterator 或最终结果。

## 3. IPC 与 wire

默认 Python API 使用 ZMQ 的 Python object channel；headless/SMG 使用有边界验证的
msgpack wire。`PortArgs` 集中分配 scheduler input/output、control、distributed init
等端口。

`EngineCoreClient` 负责主进程 request/output IPC；worker 的
`EventLoop._init_interprocess_comm()` 或 `_init_msgpack_transport()` 建立对应 socket。
DP rank 在 handshake 中带 engine identity，确保 gateway 能把输出路由回正确实例。

wire 接收到 validation error 时不会静默丢请求：request handler 把它构造成预完成的
abort state，使客户端得到 terminal response 而不是挂起。

## 4. Worker 初始化

`run_event_loop()` 创建
[`runtime/engine/event_loop.py::EventLoop`](../python/tokenspeed/runtime/engine/event_loop.py)。
主要初始化顺序：

1. distributed mapping、device、TP/DP/EP process groups；
2. `ModelConfig`、tokenizer/EOS 与 `RequestHandler`；
3. `ModelRunner` 加载模型与权重；
4. `create_attn_components()` 创建 attention backend 与 cache；
5. `ModelExecutor` 创建 sampling、grammar、speculative、runtime state 与图执行器；
6. 从 cache arena contract 构造 C++ `SchedulerConfig`；
7. 按配置创建 Host L2、PD、EPD、multimodal、pause/memory controller；
8. graph capture/autotune/warmup 完成后通过 pipe 回报 ready。

## 5. Attention 与 cache 创建

[`layers/attention/registry.py::create_attn_components()`](../python/tokenspeed/runtime/layers/attention/registry.py)
根据架构/config 选择 cache family：MHA、MLA、DSA、MSA、Qwen GDN、Inkling、Kimi
K3 或 DeepSeek V4。

当前统一路径：

```text
prepare_cache_setup()
  → merged CachePoolSpec
  → create_cache_arena()       # target + draft 唯一 allocation/contract
  → create_cache_pool(target)
  → create_cache_pool(draft)   # 可选，同一 arena 的 continuation window
```

recipe 运行 `layers → group → pack → bind`，field dtype/shape/stride 都写进 memory
plan。scheduler 几何只从 `pool.arena.runtime_contract` 读取。详细见
[Cache 文档](cache.md)。

## 6. C++ Scheduler 配置

[`engine/scheduler_utils.py`](../python/tokenspeed/runtime/engine/scheduler_utils.py)
完成边界转换：

- `scheduler_cache_geometry_from_pool()`：P、parent 数、token capacity；
- `pool_to_cache_groups()`：Python `CacheGroupSpec` → C++ `CacheGroupConfig`；
- `aligned_max_scheduled_tokens()`：对齐 recurrent-state checkpoint grain；
- `make_config()`：role、batch/token budget、L2/L3、prefix/PD/overlap 等。

`tokenspeed_scheduler.Scheduler` 在构造时统一 `Validate()`。它拥有 request FSM、prefix
index、physical block allocator、Host tier 与 per-group block tables。

## 7. 新请求进入 worker

每轮 `_process_new_requests()` 从 IPC 批量读取。`RequestHandler`：

1. 区分 generate、abort、flush、pause、profile、weight/memory control 等消息；
2. 对生成请求构造 C++ `RequestSpec(request_id, tokens, max_new_tokens)`；
3. 构造 Python `RequestState`，保存 sampling、grammar、output 与统计状态；
4. 限制 `max_new_tokens`，保证 prompt+output 不超过 worker max request length；
5. PD 请求附带 `BootstrapInfo`；
6. 交给 `scheduler.submit_requests()`。

EPD multimodal prefill 还需等待异步 embedding transfer；non-overlap loop 在固定位置
`_drain_ready_epd_embeddings()`，确保所有 rank collective 顺序一致。

## 8. 非 overlap event loop

`EventLoop.event_loop()` 每轮按以下顺序：

1. `_process_new_requests()`；
2. drain EPD embeddings；
3. `_commit_cache_results()` 将 L2 完成事件回灌 scheduler；
4. 若暂停则执行 idle/drain；
5. `scheduler.next_execution_plan()`；
6. 发布 scheduler KV events；
7. `model_executor.zero_cache_pages(plan.pages_to_zero)`；
8. `_submit_cache_ops(plan)` 提交 Host L2 op；
9. 取本轮 `Forward.Batch`，执行 DP rank 同步；
10. 收集 sampling params 与 grammar state；
11. `_dispatch_forward()`；
12. `_commit_forward_results()` 更新 Python request state；
13. poll PD transfer 并生成事件；
14. `advance_forward()` 把 extend/finish/abort/PD event 回灌 C++ FSM；
15. 更新 metrics 与 pause drain。

`pages_to_zero` 保留 group id；hybrid/V4 pool 只清零该 group 的 field byte ranges，并
用 CUDA event 保证 forward 前完成。

## 9. Overlap loop

`event_loop_overlap()` 允许当前 GPU forward 与上一轮 CPU result commit 重叠：

```text
iteration n:
  launch forward(n)
  commit result(n-1)
```

关键约束：

- 默认 stream 在写 table 前等待 execution stream 的上一轮读；
- eager grammar 必须先 commit 上一 token、推进 matcher，再填本轮 grammar mask，因而
  对 grammar batch 暂时牺牲 overlap；
- DP rank 即使无本地请求也执行 idle forward/collective；
- Prefill/EPD 强制 non-overlap，因为 embedding drain 与 PD send 只接入普通 loop；
- `overlap_schedule_depth` 当前只支持 0 或 1，并参与 cache sizing/reservation。

## 10. `ExecutionPlan` 与 forward 输入

C++ plan 可包含：

- 一个 `Forward.Batch`；
- `Cache.LoadBackOp` / `WriteBackOp`；
- `pages_to_zero`。

Forward batch 携带 request ids、request-pool indices、input ids、prefill/decode lengths、
extend prefix、每组 generic block tables 与 mode 信息。

[`scheduler_utils.block_tables_from_forward_op()`](../python/tokenspeed/runtime/engine/scheduler_utils.py)
解析 nanobind ndarray。Attention backend 随后把 group block table 映射为 kernel page
table或 state block table；`InputBuffers`、runtime state 与 draft staging 保存 GPU 侧
执行输入。

## 11. `ModelExecutor`

[`runtime/execution/model_executor.py`](../python/tokenspeed/runtime/execution/model_executor.py)
将 plan 变成一次 GPU step：

1. `execute_forward_op()` 更新 request/runtime/cache table；
2. 构造 `ForwardContext`，填 mode、length、positions、multimodal 与 cache metadata；
3. `_run_target_forward()` 经 `ModelRunner` 调模型；
4. 有 drafter 时执行 proposal/verify、维护 draft page staging 与 accepted tokens；
5. 构造 logits/sampling info，应用 grammar/logits processor；
6. sampling backend 采样并异步拷回必要结果；
7. 返回 `ModelExecutionResult`，由 event loop commit。

执行后端有三类：

- eager；
- `CudaGraphWrapper`：decode/verify batch graph；
- `PrefillGraph`：可打断 extend/prefill graph。

初始化时先 autotune，再 capture；workspace pool 在 capture 前 freeze，避免 serving 时
新增 scratch 改变地址。

## 12. Prefill 与 decode

### Prefill / extend

Scheduler 根据 prefix hit、chunk budget、state checkpoint grain 和 context limit切 prompt。
中间 chunk 的 sampled token会被丢弃，grammar matcher也不能前进；最后 chunk 产生首个
可见 token。完成的 cache boundary 被发布到 prefix index，后续请求可复用。

### Decode

每个 active request 提供少量 input token；speculative mode 可生成多个 draft token 再
由 target verify。accepted count 通过 `ExtendResult` 回灌 scheduler，只有 commit 后
request token/container/cache progress 才前进。

## 13. Logits、sampling 与 grammar

模型 forward 产出 hidden/logits；ModelExecutor 按请求 sampling params 处理 temperature、
top-k/top-p、penalty、seed 等。sampling backend 由 registry 选择并维护每请求 RNG。

Grammar 有 eager 与 capturable 两条路径：

- capturable grammar 可在 graph 中通过 hostfunc 推进并填 mask；
- eager grammar 在 overlap loop 中必须先 commit previous result；
- chunked prefill 中间 chunk 的 `advance_mask=False`。

logprob、top-logprob、指定 token logprob 与 hidden-state 由 output processor按请求选项
组装，不要求所有请求都承担相同输出开销。

## 14. 结果回传

`_commit_forward_results()`：

- 把 sampled/accepted token 加入 `RequestState`；
- 更新 stop/eos/max token/abort finish reason；
- 更新 grammar、logprob、speculative 与 request stats；
- 构造 scheduler `ExtendResult` / `Finish` / `Abort`；
- 发送增量输出到主进程。

`AsyncLLM` 的 output loop 按 request id 找到 stream state，增量 detokenize并处理 finish；
streaming 调用逐条 yield，非 streaming 调用累积到 terminal response。

## 15. PD 路径

Decode scheduler 先分配 destination block，PD receiver 据此构造 manifest；Prefill 执行
prompt 后把 cache 写入同一 plan 语义下的 Decode arena，并回传 bootstrap token。成功
后 `RemotePrefillDoneEvent` 把 C++ request 转入本地 decode。详见
[PD 分离](pd-disaggregation.md)。

## 16. 读代码顺序

1. [`runtime/entrypoints/engine.py`](../python/tokenspeed/runtime/entrypoints/engine.py)
2. [`runtime/engine/async_llm.py`](../python/tokenspeed/runtime/engine/async_llm.py)
3. [`runtime/engine/request_handler.py`](../python/tokenspeed/runtime/engine/request_handler.py)
4. [`runtime/engine/event_loop.py`](../python/tokenspeed/runtime/engine/event_loop.py)
5. [`runtime/engine/scheduler_utils.py`](../python/tokenspeed/runtime/engine/scheduler_utils.py)
6. [`tokenspeed-scheduler/csrc/scheduler/`](../tokenspeed-scheduler/csrc/scheduler/)
7. [`runtime/execution/model_executor.py`](../python/tokenspeed/runtime/execution/model_executor.py)
8. [`runtime/execution/model_runner.py`](../python/tokenspeed/runtime/execution/model_runner.py)
9. 对应 model、attention backend 和 cache pool
