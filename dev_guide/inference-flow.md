# Inference Flow
##### commit-id: 858a684af9d3b1b0330aa028e22ed2c83d9606c7
本文按代码路径总结 TokenSpeed 的一次生成请求从入口到输出的完整流程。
重点覆盖 `tokenspeed serve` 服务模式和 Python `Engine` API 共用的核心
runtime 路径。

## 总览

一次推理请求大致经过以下阶段：

```text
tokenspeed serve / ts serve
  -> SMG gateway (默认 passthrough routing) / control_server HTTP sidecar
  -> Engine / AsyncLLM (+ 可选 RL weight-transfer 控制面)
  -> InputProcessor tokenization
  -> EngineCoreClient.send_to_scheduler (ZMQ PUSH)
  -> scheduler subprocess: RequestHandler + C++ Scheduler
  -> EventLoop gets execution plan
  -> ModelExecutor builds ForwardContext and launches forward_step
     (decode CUDA graph / breakable prefill graph)
  -> model forward + logits processor + sampling backend (+ NaN guard)
  -> OutputProcesser emits scheduler events and BatchTokenIDOut
  -> AsyncLLM inline detokenizer + per-request collector
  -> stream or final response
```

当前实现已经把 detokenization 内联到 `AsyncLLM` 主进程里执行，没有独立的
detokenizer 子进程（`python/tokenspeed/runtime/entrypoints/engine.py:583`
的注释明确说明 detokenizer runs inline inside AsyncLLM）。

## 1. 服务入口

CLI 入口在 `python/tokenspeed/cli/__main__.py`。`tokenspeed serve` 不直接解析
所有 engine 参数，而是把未知参数留给 SMG orchestrator：

- `main()` 注册 `serve` 子命令：`python/tokenspeed/cli/__main__.py:67`
- `_serve()` 调用 `run_smg_from_args()`：`python/tokenspeed/cli/__main__.py:27`
  （实际调用在 `:30`）

此外 `__main__.py` 还注册了 `merge-traces` 子命令（PR #734，handler `_merge_traces`
在 `:45`，`add_parser` 在 `:86`，dispatch 在 `:111`），把同一 scheduler 进程内
Proton + VizTracer 各自导出的 trace 按 `baseTimeNanoseconds` 对齐到统一时间轴
（实现见 `python/tokenspeed/cli/trace_merge.py`）；它是离线 profiling 工具，不在
推理请求热路径上。

`run_smg()` 位于 `python/tokenspeed/cli/serve_smg.py:544`，它的启动顺序是：

1. 安装信号 handler，分配本地 engine gRPC 端口。
2. `_add_rl_control_port(engine_args)` 确保 engine argv 里带上
   `--rl-control-port`，并返回 in-engine RL 控制面的 `rl_control_url`。
3. `spawn_engine(...)` 启动 TokenSpeed engine 子进程并用 gRPC `wait_grpc_serving`
   探测 ready。
4. `spawn_gateway(...)` 启动 SMG gateway 并用 HTTP `/readiness` 探测。
5. `_start_control_server(...)` 启动 control HTTP sidecar，默认端口是 gateway
   端口加 1，并把 `rl_control_url` 转发给它。
6. 等待 engine / gateway / 终止信号其一，按 gateway-first 顺序优雅关停。

对应代码：

- RL 控制端口注入 `_add_rl_control_port(...)`：
  `python/tokenspeed/cli/serve_smg.py:432`（调用点 `:574`）
- engine 启动：`python/tokenspeed/cli/serve_smg.py:576`
- gateway 启动：`python/tokenspeed/cli/serve_smg.py:589`
- ready 输出：`python/tokenspeed/cli/serve_smg.py:604`
- control server 启动：`python/tokenspeed/cli/serve_smg.py:610`（sidecar ready
  输出在 `:619`）

`ts serve` 现在默认给 SMG gateway 注入 `--policy passthrough` 路由策略
（PR #491）。SMG 默认的健康度感知路由对单 backend 会返回
`NotImplementedError`，passthrough 策略把每个请求直接转发给唯一的健康 worker；
若 operator 显式传了 `--policy` 则保留其值：

- 默认策略常量 `DEFAULT_SMG_POLICY = "passthrough"`：
  `python/tokenspeed/cli/serve_smg.py:70`
- 注入逻辑 `_gateway_args_with_default_policy(...)`：
  `python/tokenspeed/cli/serve_smg.py:152`，由 `_gateway_args_with_defaults(...)`
  在 `:426` 调用

`ts serve` 还会给若干模型族注入默认 tokenizer/model parser
（`_args_with_default_model_parsers(...)` 在 `:378`，覆盖 DeepSeek-V4 / GLM /
Inkling / Kimi K3 等，例如 `_is_kimi_k3_model` 在 `:301`）。

旧的 `python/tokenspeed/runtime/entrypoints/http_server.py` 已经被删除并拆分成三
个文件（PR #546）。现在有两层 HTTP 面，跑在两个不同进程里：

**A. sidecar：`control_server.py`**（跑在 orchestrator 进程，端口 `gateway+1`）。
它是旧 `http_server.py` 的直接后继，由 `build_control_server(...)` 构造
（`python/tokenspeed/runtime/entrypoints/control_server.py:531`），在
`_start_control_server` 里以 daemon thread 运行。拓扑是：

- `/get_server_info`、`/get_model_info`、`/health_check`、`/abort` 直接走 engine
  gRPC unary stub（`_stub()` 懒创建单个共享 channel/stub，`control_server.py:75`）。
- `/health`、`/generate`、`/v1/*`、`/flush_cache`、`/start_profile`、`/stop_profile`
  经 `_proxy_request(...)`（`:147`）转发到 SMG gateway。
  （注意：文件顶部的 ASCII 拓扑注释 `:26` 把 `/health` 画在 gRPC-direct 分支下，
  但实际 `/health` handler 在 `:91` 走 gateway proxy——以代码为准。）
- RL 权重同步端点（vLLM 与 SGLang 两套方言）经 `_proxy_to_rl_control(...)`
  （`:390`，`_rl_control_url` 未设置时返回 503）转发到 engine 内的 RL 控制面。
  weight-version 一族端点（`/get_weight_version` `:511`、`/model_info` `:516`、
  `/update_weight_version` `:521`，PR #672）也走同一个 RL proxy。

**B. in-engine RL 控制面：`vllm_compat_http.py` + `sglang_compat_http.py`**。这两个
router 挂在 engine 进程内的同一个 FastAPI app、同一个端口（`--rl-control-port`）
上，由 `AsyncLLM._serve_rl_control_plane`（见第 2 节）拉起。vLLM 方言由
`WeightTransferManager` 驱动，SGLang 方言直接由 `AsyncLLM` 驱动。sidecar 的 RL
路由就是 proxy 到这里。

- 拓扑说明和 gRPC stub 复用：
  `python/tokenspeed/runtime/entrypoints/control_server.py:21`（stub 在 `:75`）
- RL proxy：`python/tokenspeed/runtime/entrypoints/control_server.py:390`
- vLLM 方言 router / app：
  `python/tokenspeed/runtime/entrypoints/vllm_compat_http.py:60` /
  `build_vllm_compat_app(...)` 在 `:240`
- SGLang 方言 router：
  `python/tokenspeed/runtime/entrypoints/sglang_compat_http.py:65`
  （PR #672 新增 `/get_weight_version` `:312`、`/update_weight_version` `:318`、
  `/model_info` `:337`，权重推送端点经 `_stamp_weight_version(...)` `:80` 落 version）

端到端 RL 权重同步链路：client → sidecar `control_server.py`（端口 +1）→
`_proxy_to_rl_control` → engine RL 控制面（`--rl-control-port`，vllm+sglang
router）→ `WeightTransferManager` / `AsyncLLM`。真正的权重走带外通道
（NCCL / CUDA-IPC），HTTP 上只走元数据。

weight version（PR #672）：单一真值是 `ServerArgs.weight_version`（默认
`"default"`，`server_args.py:150`，CLI `--weight-version`）。每次权重更新时
`WeightTransferManager` 把新 version 暂存到 `_pending_weight_version`
（`weight_transfer/manager.py:84`），在 `finish_update(...)` 时才 apply（避免更新
中途 stamp 到半新权重），`_apply_weight_version` 写回
`async_llm.server_args.weight_version`。输出侧 `OutputProcessor` 在构造每个响应的
`meta_info` 时直接读这个字段写入 `meta_info["weight_version"]`
（`output_processor.py:128`）。

## 2. Engine 和 AsyncLLM 前端

Python API 或 gRPC engine 最终都会进入 `Engine` / `AsyncLLM`。

`Engine.generate()` 和 `Engine.async_generate()` 会把外部参数包装为
`GenerateReqInput`：

- sync path：`python/tokenspeed/runtime/entrypoints/engine.py:145`
  （构造 `GenerateReqInput` 在 `:183`）
- async path：`python/tokenspeed/runtime/entrypoints/engine.py:205`
  （构造 `GenerateReqInput` 在 `:236`）

`Engine.__init__()`（`:110`）做三件事：

1. 构造 `ServerArgs` 和 `PortArgs`。
2. 调用 `_launch_subprocesses()` 启动 scheduler 或 data-parallel controller。
3. 在主进程持有 `AsyncLLM`（即 `tokenizer_manager`），并用 `LLM` 包一层同步
   facade 给阻塞调用方使用。

关键代码：

- 分配 IPC 端口：`python/tokenspeed/runtime/entrypoints/engine.py:129`
- 启动子进程：`python/tokenspeed/runtime/entrypoints/engine.py:133`
- scheduler 子进程 target：
  `python/tokenspeed/runtime/entrypoints/engine.py:533`（指向
  `tokenspeed.runtime.engine.event_loop.run_event_loop`，进程在 `:532` 创建）
- 创建 `AsyncLLM`：`python/tokenspeed/runtime/entrypoints/engine.py:585`

`AsyncLLM` 是主进程里的 async frontend。它拥有：

- `ModelConfig` + `tokenizer`（多模态特征由 SMG gateway 预处理后送入）
- per-request 状态表 `rid_to_state`
- scheduler 侧 ZMQ client (`EngineCoreClient`)
- `OutputProcessor`（per-request 出端处理器，包含 inline detokenizer 分支和
  logprobs 处理器）
- `InputProcessor`（请求验证 + tokenization）
- `WeightTransferManager`（RL 在线权重同步，始终常驻）+ admission gate
  `_wt_admit`（默认 set；权重同步 pause 时 clear 以拦截新请求入队）

关键代码：

- 创建 scheduler IPC client：
  `python/tokenspeed/runtime/engine/async_llm.py:140`
- 初始化 `ModelConfig` 和 tokenizer：
  `python/tokenspeed/runtime/engine/async_llm.py:145`
- 创建 `OutputProcessor`：
  `python/tokenspeed/runtime/engine/async_llm.py:221`
- output dispatcher (`TypeBasedDispatcher`) 注册：
  `python/tokenspeed/runtime/engine/async_llm.py:223`
- 创建 `WeightTransferManager` + admission gate `_wt_admit`：
  `python/tokenspeed/runtime/engine/async_llm.py:266`（gate 在 `:196`）
- 实例化 `InputProcessor`：
  `python/tokenspeed/runtime/engine/async_llm.py:272`

RL 在线权重同步（PR #546）是 `AsyncLLM` 上新增的一整套机制，只有在 engine 带上
`--rl-control-port` 时才真正拉起 HTTP 面：

- admission gate：`generate_request()` 入口先 `await self._wt_admit`
  （`async_llm.py:284`），权重同步 pause 时阻塞新请求；未 pause 时几乎零成本。
- gate / drain 辅助方法（block/allow admission、abort/drain in-flight），
  `async_llm.py:532` 一带。
- in-engine RL 控制面：`_maybe_launch_rl_control_plane(...)`
  （`async_llm.py:730`，从 `auto_create_handle_loop()` 的 `:710` 调用）在设置了
  `rl_control_port` 时用一个 uvicorn app 同端口挂载 vLLM 与 SGLang 两套 router
  （`_serve_rl_control_plane(...)` 在 `:750`）；它故意不套
  `print_exception_wrapper`，控制面崩溃不会拖垮整个进程树。

## 3. 请求 tokenization 和 ZMQ 发送

`AsyncLLM.generate_request()` 是前端热路径：

1. `await self._wt_admit`：权重同步 pause 时在此拦截新请求（未 pause 时几乎零
   成本）。
2. `auto_create_handle_loop()` 启动一次性的后台 `handle_loop()`，并附带
   sigterm watchdog、watch_load 后台任务和（若配置）RL 控制面。
3. `InputProcessor.validate_request()` 早期拦截 embedding / 生成模型不匹配。
4. `GenerateReqInput.normalize_batch_and_arguments()` 规范 batch、rid、
   sampling params、logprob 参数。
5. 单请求走 `_tokenize_one_request()` → `InputProcessor.tokenize_one_request`，
   批请求走 `_handle_batch_request()`，并行采样请求会先发一个 prefix warmup
   请求再 fan-out N 份。
6. `_send_one_request()` 把 tokenized object 推送到 scheduler。

对应代码：

- `generate_request()`：`python/tokenspeed/runtime/engine/async_llm.py:274`
  （admission gate `await` 在 `:284`，启动后台 handle loop 在 `:280`）
- 后台 handle loop 入口 `auto_create_handle_loop()`：
  `python/tokenspeed/runtime/engine/async_llm.py:701`
- 单请求 tokenize：`python/tokenspeed/runtime/engine/async_llm.py:309`
- 发送请求：`python/tokenspeed/runtime/engine/async_llm.py:322`
- 等待响应：`python/tokenspeed/runtime/engine/async_llm.py:355`
- batch 请求 fan-out：`python/tokenspeed/runtime/engine/async_llm.py:449`

tokenization 逻辑集中在 `InputProcessor`：

- 验证：`python/tokenspeed/runtime/engine/input_processor.py:85`
- text 转 input ids：
  `python/tokenspeed/runtime/engine/input_processor.py:133`
- 预计算多模态输入和 MRoPE position（SMG 送来的 precomputed mm inputs 若未带
  `mrope_*` 则调 `compute_mrope_positions(...)` 补算，并缓存
  `mrope_position_delta_scalar`）：
  `python/tokenspeed/runtime/engine/input_processor.py:170`
- context length / max_new_tokens 检查与截断：
  `python/tokenspeed/runtime/engine/input_processor.py:194`
- reasoning-parser json_schema 包装（`_maybe_wrap_json_schema_for_reasoning`，
  用 `grammar/reasoning_structural_tag.py` 的 `structural_tag_for_reasoning_json_schema`
  把 schema 包成 xgrammar structural tag，让约束只在 response channel 生效、
  reasoning 内容保持 free-form，避免 token 0 就激活 grammar 破坏 channel 前缀）：
  `python/tokenspeed/runtime/engine/input_processor.py:220`
- `SamplingParams` normalize / verify：
  `python/tokenspeed/runtime/engine/input_processor.py:222`
- output logprob 双方言判定（vLLM 用 `sampling_params.logprobs`，SGLang 用
  `GenerateReqInput.return_logprob`），并受静态 server arg
  `enable_output_logprobs` 闸门控制：
  `python/tokenspeed/runtime/engine/input_processor.py:233` 一带(gate 在 `:238`)

前端和 scheduler 之间使用 ZMQ。socket 由 `EngineCoreClient` 持有：

- `send_to_scheduler`：`PUSH` socket on `port_args.scheduler_input_ipc_name`，
  发送 tokenized request、weight-sync、session、内存占用控制以及
  load-update watcher 消息。
- `recv_from_detokenizer`：`PULL` socket on `port_args.tokenizer_ipc_name`，
  接收 `BatchTokenIDOut`、`BatchStrOut`、`BatchEmbeddingOut` 和控制面响应。

对应代码：`python/tokenspeed/runtime/engine/core_client.py:57`（PULL socket 创建在
`:59`，PUSH socket 在 `:62`）。

## 4. Scheduler 子进程初始化

scheduler 子进程入口是 `run_event_loop()`：
`python/tokenspeed/runtime/engine/event_loop.py:1985`。

`EventLoop.__init__()`（`python/tokenspeed/runtime/engine/event_loop.py:157`）
负责把执行面搭起来：

1. 加载 target / 可选 draft `ModelConfig`。
2. 通过 `DistributedInitializer` 初始化 distributed process group。
3. 由 `create_model_runner()` 造 target / draft `ModelRunner`。
4. `create_attn_components()` 造 attention backend、KV pool、draft KV pool、
   Mamba pool 等（现在返回 8 个值——PR #804 的 two-level LCM 重构新增了第 8 个
   `cache_storage` 报告，见下文）。
5. 通过 `ModelExecutorConfig.from_server_args()` + `create_model_executor()`
   组装 `ModelExecutor`。
6. 创建 KV `MemoryExecutor`——三路分支：flat KV-cache ext build 且开
   kvstore 时用 `FlatMemoryExecutor`（暂无 L3 storage tier，且拒绝 FlatKV
   contract pools 如 Kimi-K3），pool 不支持分层时为 `None`，否则用分层
   `MemoryExecutor`（KV/Mamba host pool, L3 storage 等）。
7. 用 `make_config()` 生成 `SchedulerConfig`（含 unified block pool 的
   `paged_cache_groups` 和 `prefix_cache_adjunct`，PR #447）。若开 prefix caching，
   还会先用 `aligned_max_scheduled_tokens(...)` 把 `chunked_prefill_size` floor 到
   state-snapshot page grain（recurrent-state group 只在 chunk 页对齐时才注册
   prefix-cache 页，否则复用率归零，PR #830），再构造 C++
   `tokenspeed_scheduler.Scheduler`，并 `bind_paged_cache_scheduler(...)` 把
   pool 绑定到 C++ scheduler。
8. 创建 `PauseController` / `MemoryOccupationController`、`RequestHandler` 和
   scheduler-side `OutputProcesser`。
9. 初始化 P/D KV transfer（disaggregation 模式下）；EPD encode 流水线在此附近
   建立 `epd_admission` / `_epd_staged`。

关键代码：

- 加载 model config：
  `python/tokenspeed/runtime/engine/event_loop.py:176`
  （helper `_load_model_config` 在 `:1132`）
- 初始化分布式（`_init_distributed()` → `DistributedInitializer.initialize`）：
  `python/tokenspeed/runtime/engine/event_loop.py:187`（helper 在 `:1151`）
- 创建 model runner：
  `python/tokenspeed/runtime/engine/event_loop.py:189`
- 创建 attention/KV 组件（`create_attn_components` 调用，8 元组返回）：
  `python/tokenspeed/runtime/engine/event_loop.py:212`（返回值解包从 `:203` 起，
  `self.cache_storage` 在 `:211`）
- 创建 model executor：
  `python/tokenspeed/runtime/engine/event_loop.py:254`（`create_model_executor`
  在 `:254`，`ModelExecutorConfig.from_server_args` 在 `:245`）
- `FlatMemoryExecutor` 分支：
  `python/tokenspeed/runtime/engine/event_loop.py:335`（`FlatMemoryExecutor(...)`
  在 `:350`）
- `aligned_max_scheduled_tokens` floor（prefix-cache page grain，PR #830）：
  `python/tokenspeed/runtime/engine/event_loop.py:444`（helper 在
  `scheduler_utils.py:148`）
- scheduler config 构造（`make_config`）：
  `python/tokenspeed/runtime/engine/event_loop.py:460`
- 创建 C++ scheduler：
  `python/tokenspeed/runtime/engine/event_loop.py:505`（`bind_paged_cache_scheduler`
  在 `:506`）
- pause 控制器：
  `python/tokenspeed/runtime/engine/event_loop.py:519`；内存占用控制器：`:531`
- 创建 `RequestHandler`：
  `python/tokenspeed/runtime/engine/event_loop.py:556`
- 创建 scheduler-side `OutputProcesser`：
  `python/tokenspeed/runtime/engine/event_loop.py:570`
- P/D KV transfer（`create_kv_transfer`）：
  `python/tokenspeed/runtime/engine/event_loop.py:610`
- 初始化 scheduler 侧 ZMQ（helper `_init_interprocess_comm`，`zmq.Context` +
  recv/send socket）：
  `python/tokenspeed/runtime/engine/event_loop.py:1167`（调用点在 `:515`）

attention backend 是可插拔 registry：

- 注册表与默认选择：
  `python/tokenspeed/runtime/layers/attention/registry.py:278`（`register_backend`
  在 `:281`）
- 按 `AttentionArch` 选默认 backend `_get_default_backend_name(...)`：
  MLA → `mla`，DSA → `dsa`，MSA → `msa`，其余 → `mha`：
  `python/tokenspeed/runtime/layers/attention/registry.py:317`
- 真实创建链路 `create_attn_components(...)`：
  `python/tokenspeed/runtime/layers/attention/registry.py:932`

`AttentionArch` 现在含 MLA / MHA / DSA / MSA 四种（`DSA` 用于 sparse MLA，例如
GLM-5；`MSA` 用于 MiniMax M3 的 lightning-indexer sparse attention）：
`python/tokenspeed/runtime/configs/model_config.py:83`。

具体后端如 `trtllm_mla`、`flashmla`、`dsa`、`msa` 通过 `register_backend(...)`
自注册；注册表里的完整后端集合是 `mla` / `tokenspeed_mla`（CuteDSL MLA）/
`trtllm_mla` / `flashmla` / `deepseek_v4` / `dsa` / `msa`（MiniMax M3）/ `trtllm` /
`mha`（`mha` 模块循环注册 `mha`/`fa3`/`fa4`/`triton`/`flashinfer` 五个别名到同一个
`MHAAttnBackend`）：

- `register_backend("mla", {MLA}, MLAAttnBackend)`：
  `python/tokenspeed/runtime/layers/attention/backends/mla.py:806`
- `register_backend("trtllm_mla", {MLA}, ...)`：
  `python/tokenspeed/runtime/layers/attention/backends/trtllm_mla.py:548`
- `register_backend("flashmla", {MLA}, ...)`：
  `python/tokenspeed/runtime/layers/attention/backends/flashmla.py:745`
- `register_backend("tokenspeed_mla", {MLA}, CuteDSLMLABackend)`：
  `python/tokenspeed/runtime/layers/attention/backends/tokenspeed_mla.py:1043`
- `register_backend("deepseek_v4", {MLA}, DeepseekV4AttentionBackend)`：
  `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py:2096`
- `register_backend("dsa", {DSA}, DSABackend)`（sparse MLA 后端）：
  `python/tokenspeed/runtime/layers/attention/backends/dsa.py:567`
  （`DSABackend` 类在 `:51`）
- `register_backend("msa", {MSA}, MSAHybridAttnBackend)`（MiniMax M3 dense/sparse
  路由后端）：
  `python/tokenspeed/runtime/layers/attention/backends/msa.py:934`
  （`MSAHybridAttnBackend` 类在 `:787`，内部 sparse 分支 `MSAAttnBackend` 在 `:115`）
- `register_backend("trtllm", {MHA}, TRTLLMMHAAttnBackend)`：
  `python/tokenspeed/runtime/layers/attention/backends/trtllm.py:979`
- `mha` 五别名循环注册：
  `python/tokenspeed/runtime/layers/attention/backends/mha.py:878`

注意 `flat_groups.py` 里的 `FlatCacheGroupsMixin`（`:54`）不是独立后端，而是被
`trtllm` / `mha` / `msa` 后端复用来构造 flat per-group KV-cache block table 的
mixin，从不 `register_backend`。同样地，**hybrid linear attention 也不进注册表**：
`create_attn_components` 在检测到 GDN 架构（Qwen3.5，`_HYBRID_GDN_ARCHITECTURES`
在 registry.py:289）或 KDA 架构（Kimi K3，`_HYBRID_MLA_KDA_ARCHITECTURES` 在
`:300`）时强制走 `_create_hybrid_linear_attn(...)`（`:449`，调用点 `:1324`），产出
`HybridLinearAttnBackend`（`backends/hybrid_linear_attn.py:2818`）。Inkling 后端
（`InklingAttnBackend`，`backends/inkling.py:180`）也不进注册表，而是在
`create_attn_components` 里由 `_wrap_inkling_backend(...)`（registry.py:783，调用点
`:1371`）把一个 dense MHA backend 包一层 sconv state pool，见第 9 节。

### LCM two-level KV-cache 分配（PR #804 "jenga two level allocation"）

hybrid / flat-state 模型的 KV-cache 分配从旧的 flat-slab 体系（已删除的
`kv_cache/flat_hybrid.py`、`kv_cache/flat_state_slabs.py`、`configs/hybrid_cache_plan.py`、
`configs/flat_memory_plan.py`）重构为统一的 **LCM（logical cache memory）two-level**
体系。核心概念：

- 一个 **LCM block**（也叫 parent）是共享物理 arena 的唯一分配 / 调度单位，固定为
  `_LOGICAL_BLOCK_TOKENS = 128` tokens（`lcm_setup.py:55`）。
- **two-level** = 一个 parent LCM block 里按每个 cache group 各自的
  `cache_blocks_per_lcm_block`（packing 因子）容纳固定整数个 **per-group 子 cache
  block（page）**。所以分配一个 parent 就同时为**所有** group（history KV pages、
  recurrent/conv state pages、draft history 等，各 group 的 packing 可不同）预留好
  对应数量的 page。arena 保留 parent 0 作为永不调度的 null LCM block。
- 相比旧 flat-slab（history 与 state 各一 slab），LCM 把异构的 history + linear
  attention state 打包进**一块 budget 大小的 arena**（`LcmCachePool` 持有一个扁平
  `uint8` backing，按 field 发 strided view），几何由单一 `cache_budget_bytes` 精确
  推导，colocated draft cache 复用同一套 parent 几何。

哪些模型走 LCM（`registry.py:1004`-`1016`，由 `lcm_family` 字符串决定）：

- `use_lcm_gdn` → `"qwen_gdn"`（Qwen3.5 GDN，MHA base）：需
  `is_hybrid_gdn and has_flat_state and flat_kvcache`。
- `use_lcm_k3` → `"kimi_k3"`（Kimi K3 KDA，MLA base）：需 `is_hybrid_mla_kda and
  flat_kvcache`。
- `use_lcm_inkling` → `"inkling"`（Inkling ShortConv，MHA base）：需 `is_inkling
  and flat_kvcache`。
- 若 `has_flat_state and flat_kvcache` 但无 family 匹配，直接 raise。

关键代码 / 类：

- `prepare_lcm_setup(...)`：`lcm_setup.py:578`；`create_lcm_pool(...)`：`:619`；
  `LcmPoolSpec`：`:60`。
- `LcmMemoryPlan`：`configs/lcm_memory_plan.py:81`；`plan_lcm_fields(...)`：`:217`。
- layout recipes（`configs/lcm_layouts.py`）：`mla_history_lcm_fields:28` /
  `draft_history_lcm_fields:55` / `qwen_gdn_lcm_fields:154` / `inkling_lcm_fields:224` /
  `kimi_k3_lcm_fields:357`。
- pool 类：`LcmCachePool`（base）`kv_cache/lcm.py:31`；`LcmMHATokenToKVPool`
  `kv_cache/lcm_mha.py:44`；`LcmMLATokenToKVPool` `kv_cache/lcm_mla.py:40`。

第 4 节 `create_attn_components` 的第 8 个返回值 `cache_storage`（由
`_cache_storage_report(...)` 在 `registry.py:195` 构造、`:1561` 调用，随 8 元组在
`:1569` 返回）只有跑了 LCM setup 时非 `None`：它是一个描述**实际**分配字节 / token
容量 / LCM 几何的报告 dict（`configured_cache_bytes`、`allocated_cache_bytes`、
`physical_token_capacity`、`capacity_source`、`geometry`），并校验分配 ≤ budget、
几何容量与 pool.size 一致。它被挂到 `event_loop.cache_storage`、随 scheduler ready
payload 上报，并由 DP controller 汇总。注意旧文档里 GDN heterogeneous block_size
运行时改写 `server_args.block_size` / `config.page_size` 的 `equalized_block_size`
路径已随本次重构删除——几何现在完全由 `LcmMemoryPlan` 推导。

## 5. 请求进入 scheduler

`RequestHandler` 在 scheduler 子进程里接收前端发来的对象：

- 仅 attn_tp_rank 0 通过 ZMQ non-blocking drain 收消息；
- 多 rank 时把对象 broadcast 到其他 TP rank；
- `sync_shm_features(...)` 同步多模态共享内存特征；
- 然后把对象统一交给 `process_requests(...)` 分发到生成 / 控制路径。

关键代码：

- 接收和 TP broadcast（`recv_reqs`）：
  `python/tokenspeed/runtime/engine/request_handler.py:166`
- 控制请求和生成请求 dispatch（`process_requests`）：
  `python/tokenspeed/runtime/engine/request_handler.py:193`
- 真正构造 spec / state（`handle_generate_request`）：
  `python/tokenspeed/runtime/engine/request_handler.py:264`

`TokenizedGenerateReqInput` 在这里转为两类对象：

- C++ scheduler 使用的 `RequestSpec`，只含 `request_id` 和 prompt tokens。
- Python output side 使用的 `RequestState`，包含 sampling params、eos、
  stream、logprob、多模态状态、grammar 状态等。

代码：

- `make_spec(...)`：`python/tokenspeed/runtime/engine/scheduler_utils.py:193`
  （调用点 `request_handler.py:271`）
- `RequestState.from_recv_req(...)`：
  `python/tokenspeed/runtime/engine/generation_output_processor.py:161`
  （调用点 `request_handler.py:275`）

新请求真正提交给 C++ scheduler 的位置在 `_process_new_requests()`：

- 收 recv_reqs 并扫除过期 abort 标记。
- 处理 abort / pause / grammar queue 和 pause-buffered specs。
- EPD 模式下把请求先 `epd_admission.stage(...)` 到 `_epd_staged`，暂不入
  scheduler，其 P→D sender 注册也延后。
- 对可入队请求调用 `output_processor.register(...)` 和（如有）
  `pd_kv_transfer.register(...)`。
- 可选查询 L3 KV storage hit pages，写入 `spec.rolling_hashes` /
  `storage_hit_pages`。
- 最终 `self.scheduler.submit_requests(admitted_specs)`；如果当前处于
  pause，则 buffer 起来等 resume。

对应代码：`python/tokenspeed/runtime/engine/event_loop.py:1215` 到
`:1392`。

## 6. 调度循环

普通调度循环 `event_loop()` 的每轮流程：

1. `_process_new_requests()` 收新请求并提交 scheduler。
2. `_drain_ready_epd_embeddings()`（EPD）：把异步 embedding 接收已完成的请求
   放行入队；顺序紧跟 `_process_new_requests()` 以保持 TP collective ordering。
3. `_commit_cache_results()` 提交异步 KV cache 操作结果。
4. 若被 pause，则走 `_paused_idle_step()`；否则
   `scheduler.next_execution_plan()` 取下一轮执行计划。
5. `_publish_scheduler_kv_events()` 发布 scheduler KV 事件。
6. `_handle_flat_oom_terminals(execution_plan)`（flat scheduler）：把永远塞不进
   flat pool 的请求终止并向客户端报 abort（radix build 上是 no-op）。
7. `zero_flat_cache_pages(...)`（flat KV）：清零本轮 scheduler 新分配的 flat cache
   page，返回一个 `flat_cache_zero_event` 供 `_dispatch_forward` 在 PD decode 侧同步。
   PR #804 后 zeroing 是 **group-aware（two-level）** 的：payload 优先取
   `execution_plan.flat_pages_to_zero`（per-group child pages），旧单层 scheduler 回退
   到 `flat_page_ids_to_zero`（全局 page-id 向量）；pool 侧对应 `zero_new_pages`
   （Mapping）或 `zero_pages`（page-id 列表），只有声明 `flat_kv_requires_page_zeroing`
   的 pool 缺 payload 时才 fail loudly。
8. `_submit_cache_ops(execution_plan)` 提交 L2/L3 cache load/write/prefetch。
9. 从 plan 里取 `forward_op` 并刷新 mamba retract 状态。
10. DP 模式下做 `_dp_sync_and_check(...)`，必要时执行 idle forward 保证集合
    通信对齐（并顺带算 prefill-graph 需要的 `all_extend` gate）。
11. gather sampling params 和 grammar state。
12. `_dispatch_forward(...)` 执行模型 forward。
13. 若有结果，`_commit_forward_results(...)` 后处理输出。
14. 收集 PD events 并最终 `advance_forward(self.scheduler, request_changes)`
    推进 C++ scheduler FSM；同时 `pause.maybe_finish_drain(...)` 处理延迟
    pause 应答。

代码入口：`python/tokenspeed/runtime/engine/event_loop.py:1720`。

overlap 调度循环 `event_loop_overlap()` 会先发射当前 forward，再同步和处理
上一轮 forward 的结果，以减少 GPU 空档：
`python/tokenspeed/runtime/engine/event_loop.py:1853`。eager grammar 场景
为了让 matcher 状态一致，会在派发当前 forward 之前先 commit 上一轮结果；EPD
prefill 会 assert 走非 overlap 循环。

是否启用 overlap 由 `should_use_overlap_schedule(...)` 决定；它现在只看两个入参：
显式传入的 `disable_overlap_schedule`，以及 `disaggregation_mode`——只有在
`disable_overlap_schedule` 为真、或 disaggregation 模式是 `prefill` / `encode`
时才禁用 overlap（prefill drain + KV send 只跑非 overlap 循环，encode 无 LM 循
环）：`python/tokenspeed/runtime/engine/scheduler_utils.py:327`。该结果在
`EventLoop.__init__`（`:192`）算好存为 `self.use_overlap_schedule`，最终选择在
`run_event_loop` 里完成：`python/tokenspeed/runtime/engine/event_loop.py:2078`。

## 7. Forward 派发

`_dispatch_forward()` 是 scheduler 和模型执行之间的分界点（其 docstring 明确
列出四条 path）。进入分支前会先 `update_block_table(forward_op)` 并取好
`_get_multimodal_context_for_forward(forward_op)` 供各 path 使用：

- Path 1（非 disaggregation）：直接更新 block table、reset valid cache length，
  调用 `ModelExecutor.execute_forward_op_with_log(...)`，返回 `(results, None)`。
  `kv_transfer is None` 时普通 decode（无 EXTEND）也归到这里。
- Path 2（decode 节点 + EXTEND）：触发远端 KV receive 而不跑 forward，返回
  `(None, None)`；开启 flat-KV PD 时会先 `.synchronize()` 等
  `flat_cache_zero_event` 再发布 KV destination manifest。
- Path 3（prefill 节点，无 EXTEND）：把 KV 发送给 decode side，返回
  `(None, None)`。
- Path 3b（disaggregation decode 节点普通 forward）：`DisaggDecodeExecutor` 下的
  decode-batch 正常 forward 分支。
- Path 4（prefill 节点 + EXTEND）：跑 prefill forward，并把 first token
  hand off 给 KV transfer（EPD 下先 assert embedding 已收到），返回
  `(results, on_first_token)`。

代码：`python/tokenspeed/runtime/engine/event_loop.py:885`。

`_commit_forward_results()` 会调用 scheduler-side `OutputProcesser` 做输出
解析，然后返回要喂回 C++ scheduler 的 forward events：
`python/tokenspeed/runtime/engine/event_loop.py:1395`。

## 8. ModelExecutor 内部

`ModelExecutor` 负责一次 GPU forward step 的完整准备和执行：

- 持有 `ModelRunner`、attention backend、KV pool、sampling backend；
- 维护 `req_to_page`、`InputBuffers`、`RuntimeStates`；
- 创建 `NanGuard`（per-request 数值损坏隔离，按 `enable_nan_detection` 开关）；
- 可选创建 speculative drafter：由 `_get_drafter_impl(spec_algo, model)`（`:92`，内
  含本地 `DRAFTER_MAPPING = {"EAGLE3": Eagle, "MTP": Eagle, "DFLASH": DFlash}`）挑
  选实现——`EAGLE3` 用 `Eagle`（逐步自回归起草，共享 target 的 embed / lm_head），
  `DFLASH` 用独立的 `DFlash` block drafter（自带 draft 权重、单次前向掩码填充整块、
  需 target 实现 `set_dflash_layers_to_capture(...)` 捕获指定层 hidden states）；
  `MTP` 分两种：Eagle 式 MTP（如 DeepSeek）仍走 `Eagle`，而 Inkling 的 vanilla
  多层 MTP（draft model 是 `InklingForConditionalGenerationNextN`）走 PR #730 新拆
  出的 `Mtp` drafter（`drafter/mtp.py`）；
- 可选创建 `CapturableGrammarExecutor`（CUDA）或 `EagerGrammarBuffers`；
- 注册 attention backend runtime；评估 DP-sampling 拓扑；
- 创建 `CudaGraphWrapper` 包裹 `_forward_step()`（decode graph），以及
  `PrefillGraph`（prefill/extend 的可打断 CUDA graph，见下文）；
- 对多模态模型，按模态注册 encoder CUDA graph wrapper 字典
  (`make_encoder_cudagraph_wrappers` → `encoder_graph_wrappers`，image / video
  共用一套预算图)。

关键代码：

- 初始化 executor 状态（`__init__`）：
  `python/tokenspeed/runtime/execution/model_executor.py:265`
- 创建 `NanGuard`：
  `python/tokenspeed/runtime/execution/model_executor.py:380`
- speculative drafter（`_get_drafter_impl` + 本地 `DRAFTER_MAPPING` 在 `:92`／
  `:97`，实例化在 `:386`／`:387`，DFlash 层捕获绑定在 `:431`）：
  `python/tokenspeed/runtime/execution/model_executor.py:386`
- grammar runtime（`CapturableGrammarExecutor` 在 `:456`，`EagerGrammarBuffers`
  在 `:463`）：
  `python/tokenspeed/runtime/execution/model_executor.py:445`
- DP sampling 配置：
  `python/tokenspeed/runtime/execution/model_executor.py:500`
- 创建 `CudaGraphWrapper`：
  `python/tokenspeed/runtime/execution/model_executor.py:555`
- 创建 `PrefillGraph`：
  `python/tokenspeed/runtime/execution/model_executor.py:578`
- 多模态 encoder CUDA graph wrapper 字典：
  `python/tokenspeed/runtime/execution/model_executor.py:600`

`execute_forward_op_with_log(...)` 是带 log 的入口
(`python/tokenspeed/runtime/execution/model_executor.py:1264`)，主体在
`execute_forward_op(...)`（`:1757`）：

1. 等待上一轮 execution stream 的状态写入。
2. `nan_guard.reset(bs)` 在 graph 外清零 per-request flag。
3. `input_buffers.fill_input_buffers(...)` 把 `forward_op` 转成 GPU tensor。
4. 计算 `ForwardMode`：`EXTEND` / `DECODE` / `MIXED` / `IDLE`（`forward_batch_info.py:32`；
   spec-decode 的 verify 已折叠进 `DECODE`，不再有独立的 target-verify mode）。
5. 对 mamba 模型，按需 reset / zero mamba 状态。
6. 为 EXTEND/MIXED 构造 `gather_ids`，只取需要算 logits 的位置。
7. 构造 `ForwardContext`，里面携带 attention backend、KV pool、req_to_page、
   batch size、num_extends、forward mode 等（现在还带 DSA 相关字段）。
8. 准备 grammar bitmask 和 sampling backend 的 per-step 状态
   (`setup_grammar_step` + `sampling_backend.prepare_step`)。
9. 调用 `self.forward_step(...)`（`CudaGraphWrapper` 实例，可走 graph 也可
   eager）。
10. `_update_runtime_state(...)` 写下一轮输入和 cache length。
11. `_snapshot_mamba_checkpoints(...)`（mamba 模型）。
12. 输出 token 在 D2H 之前于 GPU 上 in-place clamp 到 `[0, vocab)`
    （`output_tokens.clamp_(0, vocab-1)`，越界 id 会让 HF `tokenizer.decode`
    抛 `OverflowError` 拖垮整个进程树，故必须在 non-blocking 拷贝前 clamp），
    再把 output tokens / accept lengths / logprobs / NaN flags 通过 sampling
    backend 的 `get_packed_output_d2h(...)` D2H 拷回。
13. 返回 `ModelExecutionResult`（含 `copy_event`、`grammar_completion`、
    `next_input_ids`、`output_logprobs`、`output_nan_flags`）。

关键代码：

- 主体入口（`execute_forward_op`）：
  `python/tokenspeed/runtime/execution/model_executor.py:1757`
- `nan_guard.reset`：
  `python/tokenspeed/runtime/execution/model_executor.py:1794`
- 构造 `ForwardContext`：
  `python/tokenspeed/runtime/execution/model_executor.py:1915`
- sampling prep（`prepare_step`）：
  `python/tokenspeed/runtime/execution/model_executor.py:1962`
- 调用 `forward_step`：
  `python/tokenspeed/runtime/execution/model_executor.py:2019`
- 更新 runtime state：
  `python/tokenspeed/runtime/execution/model_executor.py:2063`
- on-GPU clamp + D2H 输出（`get_packed_output_d2h`）：
  `python/tokenspeed/runtime/execution/model_executor.py:2106`（clamp 在 `:2104`）
- 读取 NaN flags（`output_nan_flags`）：
  `python/tokenspeed/runtime/execution/model_executor.py:2118`

`_forward_step()` 是 decode CUDA graph 可捕获的核心
(`python/tokenspeed/runtime/execution/model_executor.py:960`)：

1. 可选调度 capturable grammar bitmask fill（`schedule_fill`）。
2. `_run_target_forward(...)` 调模型（`:747`）；prefill/extend（含 extend+decode
   mixed）若 `prefill_graph.can_run(...)` 为真则走 `prefill_graph.replay(...)`
   （`:768`），否则回退 eager `model_runner.forward`。
3. `nan_guard.audit_logits(...)`：在任何 sampling kernel 之前，按 request 标记
   NaN logits 并就地 `nan_to_num_` sanitize（`:997`）。
4. speculative decoding 时取 draft candidates。
5. capturable grammar 的 `wait_bitmask()` 同步。
6. `_run_sampling(...)` 采样或 verify（`:861`）。
7. `nan_guard.merge_oov(...)` 记录采样出的 out-of-vocab id（`:1015`），供输出侧
   终止对应 request。防 -1 sentinel / 越界 id 流到下游的 clamp 已上移到
   `execute_forward_op` 的 D2H 之前（on-GPU `output_tokens.clamp_(0, vocab-1)`，
   见上文第 12 步），`_forward_step` 内不再做 clamp。
8. 可选 `schedule_post_sampler(...)`（`:1020`）让 grammar 矩阵在下一步前更新。
9. speculative drafter 写下一轮 `future_input_map`。

`NanGuard` 本体在 `python/tokenspeed/runtime/execution/nan_guard.py`：
graph-safe、无同步，detection 用一次 fused `amax` reduction（NaN 通过 amax
传播），sanitize 用一次 `nan_to_num_`，flags 是 `[bs]` int32 buffer；禁用时
`NanGuard.create(...)` 返回 no-op 单例。被标记的 request 在输出侧以
`ABORT_CODE.NumericalError` 终止（见第 11 节）。

### Prefill / extend 可打断 CUDA graph

`PrefillGraph`（`python/tokenspeed/runtime/execution/prefill_graph.py`，PR #597 /
#611 默认开启）是 decode `CudaGraphWrapper` 在 extend 侧的对应物，底层是
`breakable_cuda_graph.py` 提供的 `BreakableCapture`：

- 一次 forward 被切成若干 segment——捕获好的 `CUDAGraph.replay` 与在
  attention / KV op 处的 eager "break" 交错（这些 op 的 metadata 数据相关、无法
  捕获）。模型用 `@break_point` 装饰其 attention / mixer 的 `forward` 来标记断点，
  `active_forward` / `current_forward_ctx` 在每次 replay 时重绑活的 `ctx`。
- `PrefillGraph` 按 padded token 数分桶（decode 按 batch size 分桶），每桶一张
  breakable graph，所有 segment 共享一个 mempool + 一条 capture stream，故显存
  ≈ 最大桶而非各桶之和。
- embedding lookup 留在 graph 外：graph 读一个静态 input-embeds buffer，replay
  时由 eager `embed_tokens` gather（text）或模型的 `multimodal_input_embeds`
  seam（多模态）填充。
- 捕获在 `__init__` 完成，借用 decode wrapper 的 stream 但用单独的 mempool
  （复用 decode pool 曾导致 illegal memory access，即 PR #667 的修复）；捕获失败
  经一次 MIN all-reduce 降级到 world-agreed eager，保证 DP/TP rank 步调一致。
  rank 0 捕获时用 tqdm 进度条显示每桶剩余显存。
- FlatKV **contract-bound** pool（Kimi-K3 的 MLA/KDA）现在也能捕获 prefill graph：
  旧的 "runtime_contract 非 None 就强制 eager prefill" 排除项已删除
  （`prefill_graph.py`），dummy batch 现在会为 state group 造一个可写 state page
  （`_dummy_flat_tables` 用 `torch.full` 填 1 而非全 0），并在 contract 存在时用
  `FlatCacheBatchMetadata.from_forward_op(...)` 构造捕获期的 flat-cache metadata
  （`prefill_graph.py:518` 一带）。

派发口在 `_run_target_forward`（`model_executor.py:747`，replay 判定在 `:768`）。
`ModelExecutorConfig` 新增 `disable_prefill_graph` / `prefill_graph_max_tokens` /
`prefill_graph_capture_sizes` 等开关。目前 `kimi_k25.py`、`kimi_k3.py` 暴露了
`multimodal_input_embeds` seam；MiniMax M3（PR #837）通过 `MiniMaxM3Model.resolve_embed`
接入 text-only prefill graph replay，其余多模态模型走 text-only replay 或 eager。

## 9. 模型 forward

`ModelRunner.forward()` 只是把 runtime 参数传给具体模型：
`python/tokenspeed/runtime/execution/model_runner.py:127`。

通用 causal LM 路径在 `BaseCausalLM`：

- 构造 transformer model、lm head、logits processor：
  `python/tokenspeed/runtime/models/base/causal_lm.py:47`
- `forward()` 调 `self.model(...)` 得 hidden states，再调 logits processor：
  `python/tokenspeed/runtime/models/base/causal_lm.py:147`

以 Qwen3 为例：

- `Qwen3Attention.forward()` 做 qkv projection、q/k RMSNorm、RoPE、
  `PagedAttention` 和 output projection：
  `python/tokenspeed/runtime/models/qwen3.py:204`
- `Qwen3DecoderLayer.forward()` 走 input RMSNorm、attention、post-attention
  RMSNorm、MLP：
  `python/tokenspeed/runtime/models/qwen3.py:273`
- `PagedAttention.forward()` 自身只 reshape K/V，然后委托给
  `ctx.attn_backend.forward(...)`：
  `python/tokenspeed/runtime/layers/paged_attention.py:65`（委托在 `:86`）

GLM-5 模型（`python/tokenspeed/runtime/models/glm5.py` 的
`GlmMoeDsaForCausalLM`（类在 `:1230`），及 `glm5_nextn.py:181` 的
`GlmMoeDsaForCausalLMNextN` MTP NextN 路径）走 DSA sparse MLA attention：indexer
计算 top-k，再交给 DSA backend 的 sparse prefill / decode kernel；runtime 侧通过
`tokenspeed_kernel.ops.attention` 的 `dsa_plan` / `dsa_prefill_topk` /
`dsa_decode_topk`（以及 backend 内的 `dsa_prefill` / `dsa_decode`）调用，kernel
实现见 `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/dsa.py`、
`.../triton/dsa_topk.py` 和 `.../flashinfer/dsa_topk.py`（旧的
`dsa_sparse_layout.py` 已删除）。GLM-5 现在也接入了可打断 prefill graph
（`@break_point` / `slice_to_real_tokens` / `current_forward_ctx`），并移除了 DSA
decode top-k 的 per-token 展开。

Kimi K3（`python/tokenspeed/runtime/models/kimi_k3.py` 的
`KimiK3ForConditionalGeneration`（类在 `:1910`；LM 主体 `KimiLinearForCausalLM`
在 `:1740`）是逐层混合注意力：`KimiLinearDecoderLayer`（`:1269`）按
`config.is_kda_layer(layer_id)` 分派——KDA 层用 `KimiLinearKDA`（`:615`，Kimi
Delta Attention，一个 short-conv + gated-delta-rule 的线性注意力子层，经
hybrid_linear_attn backend 的 KDA 分支跑，类比 Qwen3.5 的 GatedDeltaNet），其余层
用 `KimiLinearMLAAttention`（`:287`，直接复用 DeepseekV3 的 NoPE-MLA），故整体归入
`attention_arch=MLA`。视觉塔 `KimiK3Vision(MoonViTVisionPath)`（`:161`，基于
`moonvit.py` 的 MoonViT 3D encoder）经 `VisionEmbedder` 别名 splice，并暴露
`multimodal_input_embeds` seam（`:2013`）供可打断 prefill graph；MTP NextN 在
`kimi_k3_nextn.py` 的 `KimiK3ForConditionalGenerationNextN`。

MiniMax M3（`python/tokenspeed/runtime/models/minimax_m3.py` 的
`MiniMaxM3SparseForConditionalGeneration`（类在 `:1550`，forward 在 `:1658`；LM
主体 `MiniMaxM3SparseForCausalLM` 在 `:919`）也是逐层混合：`MiniMaxM3Attention`
（`:557`）对 sparse 层用 lightning-indexer（`MiniMaxM3Indexer` 在 `:500`）——
indexer 的 `index_q/index_k` 被融进 QKV GEMM（`MinimaxM3QKVParallelLinearWithIndexer`
在 `:309`，一次投影出 `[q|k|v|index_q|index_k]`），其余层是 dense MHA。它引入了新的
`AttentionArch.MSA`（第 4 节）而非 DSA，backend 是 `msa`
（`MSAHybridAttnBackend`）。视觉塔 `MiniMaxM3VisionTransformer`（`:1257`）经新名
`MultimodalEmbedder`（`:1602`）splice；无 NextN。PR #837 起 M3 接入 text-only
prefill CUDA graph（`MiniMaxM3Model.resolve_embed` 显式给出 text embedding seam）。

Inkling（TML，PR #689）是新接入的多模态 MoE 模型族，走 MHA attention arch 但形态
特殊：类在 `python/tokenspeed/runtime/models/inkling.py` 的
`InklingForConditionalGeneration`（`:1626`），MTP NextN draft 在
`inkling_nextn.py:213` 的 `InklingForConditionalGenerationNextN`。它有四个非标准
部件：(1) relative attention——不用 RoPE，pre-softmax logits 上加一个 query
相关的 per-head 因果距离 bias（`rel_mha` 算子族），softmax scale 是 `1/head_dim`；
(2) sconv——每 block 四处的残差 per-channel 因果 FIR，per-request rolling 状态存在
`InklingConvStatePool`；(3) shared-expert-sink MoE；(4) muP logits。它的 attention
后端不进 registry，而是在 `create_attn_components` 里由 `_wrap_inkling_backend(...)`
（registry.py:783）给一个 dense MHA backend 包一层 sconv state pool，产出
`InklingAttnBackend`（`backends/inkling.py:180`）。视觉/音频塔（HMLP patch encoder /
dMel）经共享的 `VisionEmbedder` 把特征 splice 进文本 embedding；paged-conv 默认下
支持 prefix caching。开 flat_kvcache 时 Inkling 的 KV/state 走 LCM 两级 arena
（`lcm_family="inkling"`，见第 4 节）。`model_config` 把它同时归入
`is_multimodal_model` 和 `is_audio_model`（`configs/model_config.py:753` / `:775`）。

多模态运行时已从 image-only 泛化为按模态注册的 encoder 抽象：核心类是
`multimodal/embedder.py:228` 的 `MultimodalEmbedder`（旧名 `VisionEmbedder` 现是
`:706` 的兼容别名，`qwen3_5.py` / `kimi_k25.py` / `kimi_k3.py` / `inkling.py` 仍用
别名，新增的 `qwen3_omni.py` / `qwen3_asr.py` / `minimax_m3.py` 直接用新名）。它按
模态调用注册的 encoder（image / video / audio，通过 `EncoderSpec` map 按
`Modality.*` 分派），把特征 splice 进文本 embedding，再把 `input_embeds` 送入语言
模型 forward。image 与 video 共用一套 encoder CUDA graph 预算图（受
`TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH` 开关控制，`flashinfer_cudnn` 后端
除外）。

encoder 现在支持 item-level data parallelism（PR #731）：由
`--mm-encoder-tp-mode` 选择——默认 `weights` 用 TP 切分 encoder 权重；`data` 则复
制 encoder 权重、把整个多模态 item 按 token 数做 LPT 打包分摊到 attention TP 组
内各 rank（`_assign_encoder_items` → `EncoderDPAssignment`），每 rank 只 encode
本地 item，再用精确尺寸的 broadcast all-gather 拼回（`_encode_data_parallel` 在
`embedder.py:483`，`_gather_encoder_outputs` 收集）。DP group / rank 由构造
`MultimodalEmbedder` 时传入的 `encoder_mapping: VisionTowerMapping` 决定，
`has_encoder_dp` 为真（group 大小 > 1）时才走 DP 路径。

以 Qwen3.5 为例：

- `Qwen3_5ForConditionalGeneration.forward()`（encode → splice → LM forward）：
  `python/tokenspeed/runtime/models/qwen3_5.py:1457`（`vision_embedder.apply`
  在 `:1479`）
- 按模态注册 encoder CUDA graph wrapper（`make_encoder_cudagraph_wrappers`）：
  `python/tokenspeed/runtime/models/qwen3_5.py:1404`
- `MultimodalEmbedder.apply()`：
  `python/tokenspeed/runtime/multimodal/embedder.py:249`

M-RoPE position 的计算集中到新的 `python/tokenspeed/runtime/multimodal/mrope.py`
（替代了旧的 `BaseMultimodalProcessor` / `processor_registry` 体系）：入口
`compute_mrope_positions(hf_config, input_ids, mm_items)`（`:235`）在未 padding 的
`input_ids` 上按 HF config + `grid_thw` 算三轴 `(mrope_positions,
mrope_position_delta)`，非 MRoPE 模型返回 `(None, None)`；Qwen3-Omni 走独立的
`_compute_qwen3_omni_mrope_positions`（`:140`）分别处理 image/audio/video。第 3 节
`InputProcessor` 就是调用这里补算 MRoPE。

## 10. Logits 和 sampling

`LogitsProcessor.forward()`（`python/tokenspeed/runtime/layers/logits_processor.py:366`）
会：

1. 根据 `gather_ids` 取 prefill 最后 token 或 decode token 的 hidden states。
2. 调 `_get_logits(...)`（`:558`）计算 lm head logits：走融合的
   `lm_head_gemm`（`_get_fused_lm_head_gemm` 在 `:147`）或回退
   `_lm_head_matmul` 里的 `torch.matmul`（`:172`）。
3. TP 场景下 vocab shard 的 all-gather：小 batch（`<= _LOGITS_AG_MAX_TOKENS`）
   走融合的 `all_gather_inner`（`:629`），大 batch 走
   `all_gather_into_tensor`（`:645`）；DP-sampling 时改走
   `swap_batch_vocab`。greedy + TP 场景还有一条 distributed-argmax 快路径：直接
   返回本地 shard、跳过 all-gather，稍后由 `_argmax()` 的 `distributed_argmax`
   在各 shard 上做 argmax（受 `_LOGITS_DIST_ARGMAX_MAX_TOKENS` 闸门控制）。该快路径
   现在还要 CuTe DSL argmax kernel 可用才走（`dist_argmax_available()`，某些 SKU 如
   H20 缺 TMA cluster launch 时回退到普通 all-gather + argmax）。
4. 裁剪到真实 vocab size。
5. 可选 softcap 和 logprob 计算。

`LogitsProcessorOutput` 现在同时承载 sampled-token logprob 和 top-k / 指定
token id 的 logprob（input 侧和 output 侧各一组），见
`python/tokenspeed/runtime/layers/logits_processor.py:71`。

sampling backend 由 `create_sampling_backend(...)`
(`python/tokenspeed/runtime/sampling/registry.py:54`) 创建，默认选择在
`_get_default_backend_name()`（`:40`）：

- NVIDIA 默认 `flashinfer`。
- 非 NVIDIA 默认 `greedy`。

注册表现在含五个后端：`flashinfer` / `flashinfer_full`（PR #280 起新增的
`triton` / `triton_full` 提供不依赖 FlashInfer 的可移植采样路径）/ `greedy`：

- `flashinfer`：`python/tokenspeed/runtime/sampling/backends/flashinfer.py:620`
- `flashinfer_full`：
  `python/tokenspeed/runtime/sampling/backends/flashinfer_full.py:499`
- `triton`：`python/tokenspeed/runtime/sampling/backends/triton.py:707`
- `triton_full`：`python/tokenspeed/runtime/sampling/backends/triton_full.py:572`
- `greedy`：`python/tokenspeed/runtime/sampling/backends/greedy.py:253`

`GreedySamplingBackend.sample()` 使用 `tokenspeed_kernel.ops.sampling.argmax`
(单步 argmax，verify 走 chain-greedy)：
`python/tokenspeed/runtime/sampling/backends/greedy.py:165`（argmax 调用在
`:180`，verify 在 `:196`）。

`tokenspeed-kernel` 的公共 `argmax(...)` 会：

1. 检查输入维度 / device / dtype，必要时回退到 `torch.argmax`。
2. 通过 `select_kernel("sampling", "argmax", ...)` 选择 kernel。
3. 找不到 kernel 时回退到 `torch.argmax`。

代码：
`tokenspeed-kernel/python/tokenspeed_kernel/ops/sampling/__init__.py:61`
（fallback 在 `:94` / `:106`，`select_kernel` 调用在 `:98`）。

kernel registry 和选择策略位于：

- `KernelRegistry`：
  `tokenspeed-kernel/python/tokenspeed_kernel/registry.py:249`
- `select_kernel(...)`：
  `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:583`

这也是 runtime 依赖 kernel 的边界：runtime 侧通过 `tokenspeed_kernel.ops.*`
公共 API 调用，不直接依赖具体第三方 kernel 包。

## 11. 输出处理和回传

GPU 输出回到 CPU 后，scheduler-side
`OutputProcesser.post_process_forward_op()`
(`python/tokenspeed/runtime/engine/generation_output_processor.py:548`) 会：

1. `add_cached_tokens(...)` 累计 cached tokens；`ModelExecutionResult.sync()`
   等 D2H 完成。
2. 处理 capturable grammar completion：必要时 host 端补齐 matcher。
3. 把 `output_logprobs` / `output_nan_flags` D2H 结果转成 list。
4. **flat-retract 跳过**：若某 extend slot 在 flat retract 后被 `RebasePrefill`
   把已生成 token 重新并回 prefill 窗口（`extend_prefix_lens[i] +
   input_lengths[i] < prefill_lengths[i]`），则 C++ 侧不欠该 slot 结果、采样出的
   token 是垃圾，直接 `continue` 跳过。
5. 更新每个 request 的 computed length、output ids、logprobs。
6. **NaN 隔离**：若某 request 的 nan flag 置位且尚未结束，则用
   `FINISH_ABORT(err_type=ABORT_CODE.NumericalError)` 终止它、只保留一个
   sanitized token、记 `record_nan_abort()` metric，并跳过 first-token
   hand-off（其 KV 不可信）。
7. 检查 EOS、stop token、stop string、grammar termination、max_new_tokens；
   spec-decode 累加 `spec_verify_ct`。
8. 生成 scheduler events：`ExtendResult` / `Finish` / `UpdateReserveNumTokens`
   （NaN-corrupted 走 `Abort`，跳过 radix-tree 写回，避免污染的 KV 被复用）。
9. 把要返回给前端的 token / decode_ids / logprobs 打包成 `BatchTokenIDOut`。
10. 通过 `send_to_tokenizer.send_pyobj(batch_id_out)` 直接推回 AsyncLLM。

`reap_finished_orphan(rid, state)`（`:443`）处理没有后续 forward op 去 reap 的
已完成请求：若是 pause 触发的 abort 且 client 还在等 stream，就
`publish_finished_at_admission` 补一个终止 finish；否则直接从 `rid_to_state`
删掉，避免 rid 泄漏。

关键代码：

- post-process 主体：
  `python/tokenspeed/runtime/engine/generation_output_processor.py:548`
- NaN 终止分支：
  `python/tokenspeed/runtime/engine/generation_output_processor.py:661`
- 构造 `BatchTokenIDOut`：
  `python/tokenspeed/runtime/engine/generation_output_processor.py:991`
- 发送回 AsyncLLM：
  `python/tokenspeed/runtime/engine/generation_output_processor.py:1027`

AsyncLLM 的后台 `handle_loop()`（`python/tokenspeed/runtime/engine/async_llm.py:797`）
从 `recv_from_detokenizer` 收到这些对象，经 `_result_dispatcher` 分发到
`OutputProcessor.handle_batch_output(...)`：

- handle loop：`python/tokenspeed/runtime/engine/async_llm.py:797`
- output dispatcher 注册：
  `python/tokenspeed/runtime/engine/async_llm.py:223`
- batch output 处理：
  `python/tokenspeed/runtime/engine/output_processor.py:110`

`OutputProcessor` 内部：

- `handle_batch_output(...)`（`output_processor.py:110`）在每个响应的
  `meta_info` 里写 `"weight_version"`（读 `server_args.weight_version`，PR #672，
  `:128`）。
- 根据 `enable_inline_detokenizer` 选择 inline 路径（懒构造
  `IncrementalDetokenizer`，把 token id 解为文本，
  `output_processor.py:204`）或 raw-token 路径（直接累积 token id）。
- 若请求要 logprob，按方言（vLLM 用 `sampling_params.logprobs`，SGLang 用
  `return_logprob`，或显式 `logprob_format`）调用
  `LogprobsProcessor.convert_logprob_style(...)` 渲染：
  `output_processor.py:143`。`LogprobsProcessor`
  (`python/tokenspeed/runtime/engine/logprobs.py:65`) 把 format-neutral 的 wire
  数组渲染成 vLLM 风格（`meta_info["logprobs"]`，`list[dict[token_id, Logprob]]`）
  或 SGLang 风格（`output_token_logprobs` 等 tuple-list）。
- 结果通过 `state.collector.put(...)`（`RequestOutputCollector`，
  `python/tokenspeed/runtime/engine/collector.py:61`）唤醒等待方，并在收尾时
  记录 metrics / 持久化 dump。

`_wait_one_response()`（`python/tokenspeed/runtime/engine/async_llm.py:355`）
被唤醒后：

- stream 请求每次 collector 有增量就 yield。
- 非 stream 请求等 `finished=True` 后 yield 最终结果并清理 `rid_to_state`。
- 如果调用方取消（`asyncio.CancelledError`），finally 会
  `abort_request(...)`（`:525`）把 `AbortReq` 发回 scheduler。

## 关键数据结构

| 数据结构 | 位置 | 作用 |
| --- | --- | --- |
| `GenerateReqInput` | `python/tokenspeed/runtime/engine/io_struct.py:65` | 外部生成请求的规范化输入（含 logprob 双方言字段）。 |
| `TokenizedGenerateReqInput` | `python/tokenspeed/runtime/engine/io_struct.py:405` | 前端发送给 scheduler 的 tokenized request。 |
| `RequestSpec` | `tokenspeed_scheduler` binding | C++ scheduler 使用的 request 描述。 |
| `RequestState` | `python/tokenspeed/runtime/engine/generation_output_processor.py:65` | scheduler-side per-request 输出状态。 |
| `ForwardContext` | `python/tokenspeed/runtime/execution/context.py:40` | 一次 model forward 的 runtime 上下文（含 DSA 字段）。 |
| `ModelExecutionResult` | `python/tokenspeed/runtime/execution/types.py:37` | GPU forward 返回的 token、长度、logprob、NaN flags 和 copy event。 |
| `NanGuard` | `python/tokenspeed/runtime/execution/nan_guard.py:65` | per-request 数值损坏检测 / sanitize / flag。 |
| `PrefillGraph` | `python/tokenspeed/runtime/execution/prefill_graph.py` | prefill/extend 的可打断 CUDA graph（按 token 数分桶；含 FlatKV contract 路径）。 |
| `LcmMemoryPlan` | `python/tokenspeed/runtime/configs/lcm_memory_plan.py:81` | two-level LCM arena 的几何计划（parent block ↔ per-group child pages 打包）。 |
| `LcmCachePool` | `python/tokenspeed/runtime/layers/attention/kv_cache/lcm.py:31` | 持有单块 LCM arena backing、按 field 发 strided view 的 flat-state KV pool 基类。 |
| `WeightTransferManager` | `python/tokenspeed/runtime/engine/weight_transfer/manager.py` | RL 在线权重同步的 vLLM 方言驱动（含 weight-version stamping）。 |
| `BatchTokenIDOut` | `python/tokenspeed/runtime/engine/io_struct.py:566` | scheduler 回传给 AsyncLLM 的 token 批输出。 |
| `ReqState` | `python/tokenspeed/runtime/engine/output_processor.py:63` | AsyncLLM frontend per-request collector 状态（含 inline detokenizer）。 |
| `LogprobsProcessor` | `python/tokenspeed/runtime/engine/logprobs.py:65` | 把 wire logprob 数组渲染成 vLLM / SGLang 方言。 |
| `EngineCoreClient` | `python/tokenspeed/runtime/engine/core_client.py:49` | AsyncLLM 持有的 scheduler 双向 ZMQ socket。 |

## 读代码建议

如果只想快速掌握主线，按这个顺序读：

1. `python/tokenspeed/cli/serve_smg.py`
2. `python/tokenspeed/runtime/entrypoints/engine.py`
3. `python/tokenspeed/runtime/entrypoints/control_server.py`（HTTP sidecar）
4. `python/tokenspeed/runtime/engine/async_llm.py`
5. `python/tokenspeed/runtime/engine/input_processor.py`
6. `python/tokenspeed/runtime/engine/core_client.py`
7. `python/tokenspeed/runtime/engine/request_handler.py`
8. `python/tokenspeed/runtime/engine/event_loop.py`
9. `python/tokenspeed/runtime/engine/generation_output_processor.py`
10. `python/tokenspeed/runtime/execution/model_executor.py`
11. `python/tokenspeed/runtime/execution/nan_guard.py`
12. `python/tokenspeed/runtime/execution/prefill_graph.py`、
    `python/tokenspeed/runtime/execution/breakable_cuda_graph.py`
13. `python/tokenspeed/runtime/execution/model_runner.py`
14. `python/tokenspeed/runtime/models/base/causal_lm.py`
15. 具体模型文件，例如 `python/tokenspeed/runtime/models/qwen3.py`、
    `python/tokenspeed/runtime/models/glm5.py`
16. `python/tokenspeed/runtime/layers/logits_processor.py`
17. `python/tokenspeed/runtime/sampling/*`
18. `python/tokenspeed/runtime/engine/output_processor.py`、
    `python/tokenspeed/runtime/engine/logprobs.py`
19. `tokenspeed-kernel/python/tokenspeed_kernel/*`

想深入特定子系统：

- speculative decoding drafter：`python/tokenspeed/runtime/execution/drafter/`
  （`eagle.py` 的 `Eagle`——EAGLE3 与 Eagle 式 MTP，`mtp.py` 的 `Mtp`——PR #730 拆出
  的多层 vanilla MTP，`dflash.py` 的 `DFlash` block drafter + `_dflash_fused_kv.py`）、
  draft model `python/tokenspeed/runtime/models/dflash.py` 和
  `inkling_nextn.py`。算法不再有独立枚举（旧的 `spec_decode/` 包已随 PR #739 删除）：
  现在是普通字符串 `ServerArgs.speculative_algorithm`，取值 `choices=["EAGLE3",
  "MTP", "DFLASH"]`（`--speculative-algorithm` 在
  `python/tokenspeed/runtime/utils/server_args.py:1561`，choices 在 `:1563`），
  drafter 实现由 `model_executor.py:92` 的 `_get_drafter_impl(...)` 选择。
- prefill / extend CUDA graph：`python/tokenspeed/runtime/execution/prefill_graph.py`
  和 `python/tokenspeed/runtime/execution/breakable_cuda_graph.py`（模型侧的
  `@break_point` 装饰用法见 `deepseek_v3.py` / `deepseek_v4.py` / `glm5.py`）。
- flat-state KV-cache（LCM two-level 分配，PR #804）：
  `python/tokenspeed/runtime/layers/attention/lcm_setup.py`、
  `python/tokenspeed/runtime/layers/attention/kv_cache/`（`lcm.py` / `lcm_mha.py` /
  `lcm_mla.py`）、`python/tokenspeed/runtime/configs/lcm_layouts.py` 和
  `lcm_memory_plan.py`（旧的 `flat_hybrid.py` / `flat_state_slabs.py` /
  `hybrid_cache_plan.py` / `flat_memory_plan.py` 已删除）。
- RL 在线权重同步：`python/tokenspeed/runtime/engine/weight_transfer/`
  （`manager.py`、`config.py`）、`python/tokenspeed/runtime/entrypoints/`
  下的 `vllm_compat_http.py` / `sglang_compat_http.py` / `control_server.py`。
- 多模态 / 视频 / 音频：`python/tokenspeed/runtime/multimodal/`
  （`embedder.py`——含 item-level encoder data parallelism、`encoder_cudagraph.py`、
  `inputs.py`、`mrope.py`）和
  `python/tokenspeed/runtime/models/qwen3_5.py`、`qwen3_omni.py`、`qwen3_asr.py`、
  `qwen3_audio.py`、`inkling.py`（含 `inkling_nextn.py` MTP NextN）、
  `kimi_k25.py`、`kimi_k3.py`（含 `kimi_k3_nextn.py`）、`minimax_m3.py`，以及共享的
  MoonViT 视觉塔 `moonvit.py`。
