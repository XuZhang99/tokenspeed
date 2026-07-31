# PD 分离（Prefill/Decode Disaggregation）详解
##### commit-id: d50bb481505c229f93594e56d5e1f8a772876451

本文详细介绍 TokenSpeed 的 **PD 分离**功能：把一次生成请求的 **prefill**（算完整
prompt 的 KV，采样出第一个 token）和 **decode**（逐 token 增量生成）拆到两组不同的
GPU 节点上跑，prefill 节点算完的 KV-cache 经 **Mooncake** RDMA 直写进 decode 节点预
先注册好的显存里，第一个 token 则随控制面 ZMQ 状态消息带过去。这样 prefill（计算密
集、大 batch）和 decode（访存密集、小 batch）可以各自独立扩缩容、用不同的并行度和
CUDA graph 策略。

代码主要在 `python/tokenspeed/runtime/pd/`（下文除特别标注外，`path:line` 锚点都相
对这个目录），跨节点事件的 FSM 部分在 C++ 调度器 `tokenspeed-scheduler/`（标注为
`tokenspeed-scheduler/...`），集成面在 `python/tokenspeed/runtime/engine/event_loop.py`
（标注为 `engine/event_loop.py`）。

本文是 `inference-flow.md`（一次请求端到端流程）在「跨节点」维度的展开，也是
`kvcache-management.md` 第 9 节「PD 跨节点 KV 传输」的完整实现细节。KV pool 的物理
布局、页分配、prefix 复用请看那两篇。

> 约定：全文用大量 `path:line` 锚点。行号会随无关改动漂移，更新本文时请逐个用
> `grep -n` / `sed -n` 核对。

## 总览

一次 PD 请求的生命周期横跨两组进程。**控制面**是 HTTP rendezvous（bootstrap 服务
器）+ ZMQ `PUSH`/`PULL` socket，传的是「对方在哪」「buffer 指针在哪」「传完了没」这
些元数据；**数据面**是 Mooncake `TransferEngine` 的**单边 RDMA 写**——prefill 是主动
写方，直接把 KV 写进 decode 侧预注册的 GPU buffer，decode 从不主动拉，只注册好指针
然后等一条 "Success" ZMQ 消息。

```text
┌─ Prefill 节点（disaggregation_mode="prefill"）──────────────────────┐
│  event_loop：_dispatch_forward Path 4 跑 prefill forward + 采样      │
│    → store_prefill_token 记下第一个 token                            │
│  DisaggPrefillExecutor：每个请求一个 MooncakeKVSender               │
│  MooncakeKVManagerPrefill：ZMQ 收 decode 的预分配 + 指针注册         │
│    → transfer_worker 线程发 KV（batch_transfer_sync_write）          │
└──────────────┬───────────────────────────────────────┬─────────────┘
   控制面：HTTP │ bootstrap（prefill PUT 自己的 ip:port）│ 数据面：RDMA
   ZMQ PUSH/PULL│（decode GET 查询）                     │ 单边写 KV
               ▼                                         ▼
┌─ Decode 节点（disaggregation_mode="decode")───────────────────────┐
│  admit 时 state.computed_length = input_length（prompt 已远端算过） │
│  DisaggDecodeExecutor：每个请求一个 MooncakeKVReceiver             │
│    → _register_kv_args 把本地 KV buffer 指针发给 prefill            │
│    → prefill() 发预分配（dst 页索引 + decode_prefix_len）           │
│  收到 Success ZMQ 消息 → RemotePrefillDoneEvent 带 bootstrap_token   │
│  event_loop：_dispatch_forward Path 3b 用第一个 token 起本地 decode │
└─────────────────────────────────────────────────────────────────────┘
```

传输后端由 `TransferBackend`（`utils.py:132`）枚举选择，只有两个成员：`MOONCAKE`
（`"mooncake"`）和 `MOONCAKE_ASYNC`（`"mooncake_async"`）——**没有 NCCL / NIXL**
（NCCL 只在 EPD 里做 embedding 行分片重组，不在 PD KV 路径）。默认走同步的
`MOONCAKE`；`MOONCAKE_ASYNC` 见 [§10](#10-mooncake_async-异步后端现状)（当前与重构
后的同步路径不同步，构造即会失败，实际是遗留分支）。

一次请求的 rendezvous 键是 **`bootstrap_room`**（一个 int）；它同时是控制面 ZMQ 每
条消息的 key、状态 FSM `request_status` 的 key，也用来选 prefill 的 DP group
（`room % dp_size`，`mooncake/receiver.py:513`）。

## 目录结构

```text
python/tokenspeed/runtime/pd/
├── factory.py            create_kv_transfer + get_kv_args：造 Executor 和 KVArgs
├── prefill_executor.py   DisaggPrefillExecutor：prefill 侧每步编排 + 事件泵
├── decode_executor.py    DisaggDecodeExecutor：decode 侧每步编排 + 事件泵
├── flatkv.py             FlatKV PD 的结构化 layout / manifest 契约（版本化、带校验）
├── transfer_plan.py      异构 TP 的传输分片规划器（PDTransferPlanner）
├── kv_events.py          （非 PD 传输）scheduler KV-cache 变更事件的 ZMQ publisher
├── utils.py              枚举、FastQueue、StepCounter、MetadataBuffers 等公共件
├── base/
│   ├── bootstrap.py      DisaggBootstrapServerBase：HTTP rendezvous（PD/EPD 共用）
│   ├── manager.py        DisaggManagerBase：engine + ZMQ 控制 socket + 状态 FSM
│   ├── mooncake_engine.py MooncakeTransferEngine：包 mooncake.engine.TransferEngine
│   └── status.py         TransferPoll：6 个状态常量
└── mooncake/
    ├── conn.py           MooncakeKVManagerBase + MooncakeKVBootstrapServer
    ├── entities.py       KVArgs / TransferInfo / TransferKVChunk / KVManagerArgs 等
    ├── prefill.py        MooncakeKVManagerPrefill：ZMQ 收线程 + transfer_worker
    ├── sender.py         MooncakeKVSender：每请求一个的发送句柄
    ├── decode.py         MooncakeKVManagerDecode：ZMQ 收线程 + 心跳 + 首 token 捕获
    ├── receiver.py       MooncakeKVReceiver：每请求一个的接收句柄 + 路由规划
    └── async_conn.py     MooncakeAsyncKVManager：异步 submit/poll 后端（遗留）
```

C++ 侧相关文件（`tokenspeed-scheduler/csrc/`）：
- `scheduler/outside_events/pd.h`：4 个 PD 事件 struct + `PDEvent` variant。
- `scheduler/outside_event_handler.cpp`：`Scheduler::handleEvent(pd::...)` 处理器。
- `fsm/pd_events.cpp` / `fsm/pd_events.h`：请求 FSM 里 PD 相关的状态转移。

## 1. 配置与启动（server_args）

PD 由几个 `disaggregation_*` server-arg 驱动，定义在
`engine/utils/server_args.py`（下面几个锚点相对
`python/tokenspeed/runtime/utils/server_args.py`）：

- `disaggregation_mode: str = "null"`（`:297`）：`"null"`（聚合式，不分离）/
  `"prefill"` / `"decode"` / `"encode"`（EPD 里的纯 vision-tower 节点）。argparse 在
  `:1874`-`:1880` 注册，`choices=["null","prefill","decode","encode"]`。
- `disaggregation_bootstrap_port: int = 8998`（`:298`）：prefill 侧 bootstrap HTTP 服
  务器端口。
- `disaggregation_transfer_backend: str = "mooncake"`（`:299`，argparse `:1898`）：
  `mooncake` / `mooncake_async`。
- `disaggregation_ib_device: str | None = None`（`:300`）：IB 设备名，支持单个
  （`mlx5_0`）或逗号分隔多个；None 走 Mooncake 自动探测。
- `disaggregation_layerwise_interval: int = 1`（`:301`）：layerwise 流式传输的层间隔。
- `pdlb_url: str | None = None`（`:302`）：PD 负载均衡器 URL，设了的话
  prefill/decode 节点启动时向它注册（`utils.py:235` 的
  `register_disaggregation_server`）。

> 注意：**没有** `disaggregation_bootstrap_host` 这个 server-arg。bootstrap host 是
> 逐请求的，随 `BootstrapInfo.bootstrap_host`（`base/bootstrap.py:37`）传，和
> `bootstrap_port`（`:38`）、`bootstrap_room`（`:39`）一起。

**模式解析与约束**在 `resolve_disaggregation()`（`server_args.py:635`）：
- `prefill`（`:637`）：强制 `enforce_eager=True`，**prefill 节点关 CUDA graph**（每个
  请求 prompt 长度不同，capture 无收益）。
- `decode`（`:640`）：prefix caching 保持可配。
- `encode`（`:646`）：纯 vision tower，关 prefix cache，拒绝 attn-DP。
- `prefill` + 非 `round_robin` 负载均衡 + attn-DP → `ValueError`（`:660`-`669`）。
- `_handle_kvstore()`（`:671`）：`decode`/`encode` 强制关 kvstore（L3 外部存储）。

**Executor 的构造**在 `event_loop.py`：`disaggregation_mode != "null"` 时
（`event_loop.py:586`）先用 `get_kv_args(...)`（`:587`）拿 KV buffer 描述符，再拼
`KVManagerArgs`（`:595`-`:610`），然后调 `create_kv_transfer(...)` 存进
`self.kv_transfer`（`event_loop.py:611`）；非-PD 节点该字段为 `None`（`:648`）。
prefill 节点接着 `_setup_pd_layerwise_transfer(interval)`（`:619` / 定义 `:652`）给
layerwise streaming 注册 `StepCounter`。

`create_kv_transfer(mode, backend, args, kv_args, gloo_group, page_size)`
（`factory.py:188`）按 mode 分派：`"prefill"` → `DisaggPrefillExecutor`（`:206`），
`"decode"` → `DisaggDecodeExecutor`（`:208`），否则 `NotImplementedError`（`:210`，
`encode` 没有 KV 传输，不走这里）。若 `kv_args.flat_layout is not None`（FlatKV），
额外要求后端是 Mooncake（`:197`-`:200`）且不能开 MLA-L1.5 / layerwise（`:201`-`:204`）。

### 1.1 KVArgs / KVManagerArgs

两个核心配置 dataclass 都在 `mooncake/entities.py`：

- **`KVArgs`**（`:39`）：**buffer 描述符**——告诉 Mooncake 要注册/传哪些显存。字段：
  `engine_rank`、`kv_data_ptrs`/`kv_data_lens`/`kv_item_lens`（`:42`-`:44`，每个连续
  KV buffer 的指针/总长/单元长）、`offsets`（`:45`，逐层 buffer 指针偏移）、`aux_*`
  （`:46`-`:48`，first-token metadata buffer）、`ib_device`/`gpu_id`、
  `target_layer_num`/`draft_layer_num`、`kv_layer_ids`/`kv_unit_lens`、`state_*`
  （`:55`-`:60`，Mamba/SSM 状态平面，`state_type="mamba"` 时用）、`flat_layout`
  （`:64`，FlatKV 时挂上结构化 layout）。
- **`KVManagerArgs`**（`:242`）：**引擎级配置**——`bootstrap_port`、`dist_init_addr`、
  `world_size`/`dp_size`/`attn_tp_rank`/`attn_dp_rank`、`is_mla_backend`/
  `draft_is_mla_backend`、`enable_metrics`/`enable_dp_attention`/
  `enable_mla_l1_5_cache`、`served_model_name`/`app_key`/`metrics_reporters`。

`get_kv_args(...)`（`factory.py:55`）负责从 KV pool 抽出这些指针：普通路径调
`token_to_kv_pool.get_contiguous_buf_infos()`（`:110`），可选拼上 draft/MTP pool
（`:119`-`:143`）和 Mamba state pool（`:153`-`:159`）；FlatKV 路径（`:63`-`:108`）改
从 `pool.get_flatkv_pd_contract()`（见 [§8](#8-flatkv-pd-契约)）拿 `(layout,
registrations)`，按原始 slab 建 `KVArgs`（拒绝 draft/mamba，`:66`-`:73`）。

## 2. event_loop 里的每步编排（四条 path）

PD 的核心分派在 `_dispatch_forward(...)`（`event_loop.py:886`，docstring 列 path 于
`:896`-`:906`）。它按 `self.kv_transfer` 的类型和这批 forward 里有没有 extend（新
prompt）请求走四条路：

- **Path 1（非 PD）**（`self.kv_transfer is None`，`:923`）：普通 forward，
  `reset_valid_cache_length` 后照常跑，返回 `(results, None)`。
- **Path 2（decode 节点 + 有 extend）**（`DisaggDecodeExecutor` 且
  `num_extends()>0`，`:943`）：新请求在等远端 KV。先
  `kv_transfer.reset_valid_cache_length(...)`（`:945`）把远端算好的 prompt 长度 seed
  进 runtime state（见 §3），FlatKV 时先 `flat_cache_zero_event.synchronize()`
  （`:951`-`956`）确保新分配的页清零完再发 destination manifest（页清零跑在 CUDA
  stream 上，而 Mooncake/GPUDirect 的写不被那个 stream 序），然后
  `kv_transfer.execute(op)` 触发 **RDMA receive**（其实是发预分配让 prefill 来写），
  返回 `(None, None)`——这一步 decode 不产出 token。
- **Path 3b（decode 节点普通 decode）**（else 分支，`:960`）：正常 decode forward，
  返回 `(results, None)`。
- **Path 3（prefill 节点，无 extend）**（`num_extends()==0`，`:981`）：prefill 都算完
  了，`kv_transfer.execute(op)` **发 KV** 给 decode，返回 `(None, None)`。
- **Path 4（prefill 节点 + 有 extend）**（else，`:985`）：跑 prefill forward。先
  `reset_valid_cache_length`（`:987`）、`kv_transfer.prepare_prefill(op)`（`:988`，
  layerwise streaming 用），forward 带 `capture_next_input_ids=True`（`:1003`）采样出第
  一个 token，返回 `(results, self.kv_transfer.store_prefill_token)`（`:1006`）——第二
  个返回值是 `on_first_token` 回调。

> 命名坑：docstring 只列 Path 1-4；代码注释里 decode 的普通 decode 分支叫「Path 3b」
> （`:961`）。prefill 的「发 KV」分支就是 Path 3（`:982`），没有「Path 3b prefill」。

**首 token 的交接**：Path 4 返回的 `store_prefill_token` 一路透传——非 overlap 循环
`:1769`-`:1783` 拿到 `on_first_token` 交给 `_commit_forward_results`（`:1395` /
透传进 `post_process_forward_op` 于 `:1411`-`:1416`）；overlap 循环丢弃它（`:1970`，
prefill 节点只跑非-overlap 循环）。真正调用在 `generation_output_processor.py:684`：
采样出 `model_output_ids` 后取 `bootstrap_token = int(model_output_ids[0])`（`:685`），
收齐 MTP `spec_candidate_ids`（`:690`-`:699`），调
`on_first_token(rid, req_pool_idx, bootstrap_token, spec_candidate_ids)`（`:701`）。
这个回调就是 prefill executor 的 `store_prefill_token`（见 §4.1）。

**register / admit**：admit 循环里（`event_loop.py:1306`-`:1336`），decode 节点先设
`state.computed_length = state.input_length`（`:1324`-`1325`，整段 prompt 标记为「已
算」，decode 永不本地重跑 prompt），再 `kv_transfer.register(rid, bootstrap)`
（`:1335`-`1336`）开一个 sender/receiver。EPD 请求推迟到 admission drain 再 register
（`:1327`-`1334`，否则 `generate_events` 会在请求还没进 C++ scheduler 时就 poll，
`requests_.at(rid)` 抛异常）。

**事件泵**：每轮循环末尾 `kv_transfer.generate_events()`（非-overlap `:1786`，overlap
`:1988`），结果进 `_process_kv_transfer_events(...)`（`:1468`，见 §5）。

**FlatKV 开关**：`self._flatkv_pd_enabled`（`event_loop.py:396`-`:399`）= mode 是
prefill/decode 且 `token_to_kv_pool.supports_disaggregation is True`；它进
`scheduler_cfg.enable_flatkv_pd`（`:484`）告诉 C++ 调度器走 FlatKV 租约/围栏语义。

## 3. Decode 侧编排（DisaggDecodeExecutor）

`decode_executor.py` 的 `DisaggDecodeExecutor`（`:44`）。每个请求一个
`MooncakeKVReceiver`，存在 `self.receivers[request_id]`。

- **`register(rid, bootstrap_info)`**（`:202`）：置本地状态 `Bootstrapping`，
  `_bootstrap`（`:65`）建 `MooncakeKVReceiver(mgr, bootstrap_addr, bootstrap_room)`。
- **`execute(op)`**（`:210`）→ `_prefill(op)`（`:159`）：给这批 extend 请求发**预分配**
  ——对每个请求算出它在本地 KV pool 里占的目标页 `kv_indices`
  （`occupied_pages[i][extend_prefix_len//page_size : prompt_end_page]`，`:185`-`:190`；
  末端砍到 prompt 逻辑结尾，reserved 的 decode 输入页不算进来），连同 `aux_index`、
  `extend_prefix_len`、可选 `mamba_indices` 调 `receiver.prefill(...)`（`:194`）。这条
  预分配经 ZMQ 发给 prefill，告诉它「把 KV 写到我这些页」。FlatKV 走 `_flat_prefill`
  （`:108`），发结构化 manifest 的 `page_ids`（不支持混合 batch，`:111`-`:113`）。
- **`reset_valid_cache_length(op, runtime_states, stream, device)`**（`:304`）：decode
  destination 从不本地跑 prompt，model executor 无法从 forward 推出序列长度；这里把
  `prefill_lengths[:num_extends]`（远端算好的完整 prompt 长度）在 execution stream 上
  `runtime_states.reset_states(...)` seed 进 runtime 行（`:317`-`336`），第一步本地
  decode 前必须做。FlatKV 还用这个序列长度选传过来的 recurrent-state 快照页。
- **`generate_events()`**（`:215`）：`poll_and_all_reduce` 把所有 receiver 的状态跨
  gloo group 做 MIN all-reduce（`:218`，TP 内保持一致），然后按状态迁移发 C++ 事件：
  - `Bootstrapping → Bootstrapped`：发 `PD.BootstrappedEvent`（`:230`）。
  - `→ Failed`：发 `PD.FailedEvent`（`:236`）并**丢掉该 receiver**（`:240`，否则一个失
    败请求每轮都重发 FailedEvent，poll 一直 Failed，把整个 conn-1 调度器卡死）。
  - `Bootstrapped → Success`：从 `kv_manager.pop_prefill_metadata(room)`（`:251`）取
    ZMQ 带来的 `bootstrap_token` + `spec_candidate_ids`，发
    `PD.RemotePrefillDoneEvent(rid, bootstrap_token)`（`:276`-`279`）把首 token 带给
    C++ FSM。MTP spec candidate 存进 `_remote_spec_candidate_ids`（`:264`），注意**不**
    在这里 pop（`:284`-`294` 那段注释：它的消费者 `pop_remote_spec_candidate_ids` 在
    `_process_kv_transfer_events` 里稍后跑；提前 pop 会让下一步 decode forward 读到未
    初始化的 `future_input_map` 尾部，触发 embedding lookup 的 CUDA illegal access）。

## 4. Prefill 侧编排（DisaggPrefillExecutor）

`prefill_executor.py` 的 `DisaggPrefillExecutor`（`:48`）。每个请求一个
`MooncakeKVSender`，存 `self.senders[request_id]`。

- **`register(rid, bootstrap_info)`**（`:370`）：置 `Bootstrapping`，`_bootstrap`
  （`:119`）建 `MooncakeKVSender(mgr, bootstrap_addr, bootstrap_room)`。
- **`execute(op)`**（`:390`）→ `_decode(op)`（`:244`）：对每个请求算出要发的页——
  `occupied_pages[i][decode_prefix_len//page_size:]`（`:279`-`282`，跳过 decode 侧已
  有的 prefix），连同 `aux_index`、`bootstrap_token`、`spec_candidate_ids`、
  `mamba_indices` 调 `sender.send(...)`（`:292`）。`decode_prefix_len` 从对应
  `TransferInfo` 取（`_decode_prefix_len`，`:173`），且必须被 `page_size` 整除
  （`:274`-`278`）。已经在做 layerwise 传输的请求跳过这里（`:262`-`270`），只补发首
  token metadata。FlatKV 走 `_flat_decode`（`:301`），要求恰好一个 identity-routed 目
  标（`:313`），发 manifest 的 `page_ids`。
- **`store_prefill_token(rid, aux_index, token, spec_candidate_ids=None)`**（`:74`）：
  §2 说的 `on_first_token` 回调。记下首 token 到 `_request_token[rid]`（`:90`）；开了
  layerwise 就立刻 `kv_manager.set_prefill_metadata(...)` 发布（`:101`-`106`）。FlatKV
  不支持 spec candidate（`:82`-85`）且要求非负 token（`:86`-89`）。
- **`prepare_prefill(op)`**（`:214`）：Path 4 里调，做 layerwise streaming——每层算完
  就发那一层的 KV 窗口（`sender.send_layerwise(...)`，`:233`），最后一层等首 token。
  未开 layerwise 或无 extend 时直接 return（`:217`）。
- **`generate_events()`**（`:393`）：和 decode 对称。`Bootstrapping → Bootstrapped` 发
  `PD.BootstrappedEvent`（`:408`）；`→ Failed` 发 `PD.FailedEvent`（`:414`）；
  `Bootstrapped → Success` 发 `PD.SucceededEvent`（`:424`）并清理该请求状态
  （`_drop_request_state`，`:126`——sender、token、spec candidate 全 pop，request_id 稳
  定不复用，不清会泄漏到引擎重启）。
- **`abort(rid, bootstrap_info)`**（`:378`）：EPD 专用——prefill 在注册 KV sender 前就
  abort 了（embedding 接收超时），调 `kv_manager.abort_room(...)` 让 dual-dispatch 的
  decode receiver 失败而不是永远等。

## 5. 事件回灌与 C++ FSM

PD 的四个事件是 **C++ 类型**，定义在
`tokenspeed-scheduler/csrc/scheduler/outside_events/pd.h`：`BootstrappedEvent`
（`:30`）、`FailedEvent`（`:35`）、`SucceededEvent`（`:40`）、`RemotePrefillDoneEvent`
（`:45`，带 `request_id` + `bootstrap_token`），合成
`using PDEvent = std::variant<...>`（`:54`）。Python 侧通过
`from tokenspeed_scheduler import PD`（`event_loop.py:33`）拿到这个 `PD` 命名空间，由两
个 executor 的 `generate_events()` 构造发出（见 §3/§4）。

> `pd/kv_events.py` **不是**这套 PD 传输事件——它是 scheduler 的 **KV-cache 变更事件**
> （`BlockStored` `:67` / `BlockRemoved` `:74` / `AllBlocksCleared` `:78`）+ 一个 ZMQ
> `EventPublisher`（`:129` / `ZmqEventPublisher` `:167` / `EventPublisherFactory`
> `:441`），供外部路由器/前缀感知负载均衡器订阅。两者别名相近，勿混。

**Python 消费**在 `_process_kv_transfer_events(events)`（`event_loop.py:1468`）：
- `PD.SucceededEvent`（prefill 侧，`:1472`）→ `output_processor.finish_prefill_request`
  （`:1476`，prefill 侧该请求 KV 已送达，可回收）。
- `PD.RemotePrefillDoneEvent`（decode 侧，`:1477`）→
  `output_processor.on_remote_prefill_done(rid, bootstrap_token)`（`:1482`）；FlatKV 再
  `finish_remote_prefill_only_request`（`:1487`）；写远端 MTP spec candidate
  （`pop_remote_spec_candidate_ids` → `write_remote_spec_candidate_ids`，`:1490`-1497）。
- `PD.FailedEvent`（`:1498`）→ `state.set_finish_with_abort(...)` 对客户端报错
  （`:1511`）；**legacy（非-FlatKV）PD 还要追加一个 `ForwardEvent.Abort()`**（`:1517`-
  1520），因为它的 C++ FailedEvent 处理器是 no-op；FlatKV 的 FailedEvent 自己原子地
  terminalize 并围栏租约资源。

**C++ 处理器**在 `tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`：
- `handleEvent(pd::BootstrappedEvent&)`（`:99`）：请求在 `fsm::Bootstrapping` 才
  `Apply(fsm::BootstrappedEvent{})`（`:104`），否则忽略。
- `handleEvent(pd::FailedEvent&)`（`:107`）：`Finished` 忽略；`WritingBack` 中抛
  `logic_error`（`:113`，不能 terminalize 进行中的 writeback）；否则 `Apply(AbortEvent)`
  （`:119`），FlatKV 构建还会 erase `pending_forward_results_` / `flat_pd_transfer_pins_`
  （`:116`-117）。
- `handleEvent(pd::SucceededEvent&)`（`:126`）：必须在 `PrefillDone` 或 `Decoding`
  （`:131`，否则 `logic_error`）。
- `handleEvent(pd::RemotePrefillDoneEvent&)`（`:148`）：请求是 `fsm::Prefilling` 时校验
  `bootstrap_token >= 0`（`:155`），FlatKV erase transfer pin（`:158`），
  `Apply(fsm::RemotePrefillDoneEvent{bootstrap_token})`（`:160`）；已是
  `PrefillDone`/`Decoding`/`Finished` 则 no-op；其它状态（还没作为 decode 目标 admit）
  抛 `logic_error("... received before decode destination admission ...")`（`:166`）。这
  就是「admit 必须先于远端 KV 完成信号」的强制点。

**FSM 转移**在 `tokenspeed-scheduler/csrc/fsm/pd_events.cpp`：
`RemotePrefillDoneEvent::operator()(Prefilling&& state)`（`:40`）把请求从 `Prefilling`
迁到 `PrefillDone`（`:46`），搬 block table / flat-cache 进度，并
`ExtendResultTokens({bootstrap_token})`（`:62`）——把远端采样的首 token 注入 decode 请
求，让它能本地起 decode。

## 6. Mooncake 后端：控制面 rendezvous

### 6.1 引擎封装（base/mooncake_engine.py）

`MooncakeTransferEngine`（`:26`）薄封装外部 `mooncake` 包：
- `__init__` import `from mooncake.engine import TransferEngine`（`:29`，装不上会提示去
  Mooncake 仓库 build），建 `TransferEngine()`（`:37`），
  `session_id = f"{hostname}:{engine.get_rpc_port()}"`（`:46`）——**session id 就是
  `host:rpc_port`**，是每次传输的目标句柄。
- `initialize(...)` → `engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device)`
  （`:74`-79）——协议是带 P2P 握手的 RDMA。
- `register(ptr, length)` → `engine.register_memory`（`:48`，把 GPU/host buffer pin 给
  RDMA）；`deregister` → `unregister_memory`（`:58`）。
- `transfer_sync(session_id, buffer, peer_addr, length)` →
  `engine.transfer_sync_write`（`:84`-91，**单边 RDMA 写**到对端地址；首次按 session_id
  建并缓存 queue pair，之后复用）。
- `batch_transfer_sync(...)` → `engine.batch_transfer_sync_write`（`:109`-118，同步路径
  的主力；缺方法时提示升级 `mooncake-transfer-engine >= 0.3.4.post2`，`:124`-128）。
- `transfer_submit_write` 返回 `batch_id`（`:139`，异步 submit）+
  `transfer_check_status(batch_id)`（`:149`，轮询）——供 `MOONCAKE_ASYNC` 用。

### 6.2 HTTP rendezvous（base/bootstrap.py + mooncake/conn.py）

`DisaggBootstrapServerBase`（`base/bootstrap.py:42`）是 PD/EPD 共用的 aiohttp HTTP 服务
器，让 decode 找到 prefill 各 rank 的 `(ip, port)`：
- `__init__`（`:57`）建 `web.Application`、`prefill_port_table:
  dict[dp_group][tp_rank_in_dp] -> {rank_ip, rank_port}`（`:64`），起 daemon 服务线程
  （`:80`-81），带超时同步启动（`run()` `:83`，`_STARTUP_TIMEOUT_SECONDS=10` `:54`）。
- 路由（`_setup_routes` `:121`）：`"*" /route`（`:122`）+ `GET /health`（`:123`，decode
  心跳探它）。
- `_handle_route_put`（`:139`，**prefill 注册**）：读
  `role/world_size/dp_size/rank_ip/rank_port/engine_rank`（`:141`-146），
  `_ingest_put_extra(data)`（`:147`）挂角色专属字段，`role=="Prefill"` 时按
  `dp_group = engine_rank // tp_size_per_dp_rank`、
  `tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank` 存表（`:159`-171）。
- `_handle_route_get`（`:181`，**decode 查询**）：哨兵 `engine_rank==-1 &&
  target_dp_group==-1` 返回并行度同步信息（`prefill_tp_size`/`prefill_dp_size` +
  `_extra_parallel_info()`，`:188`-194）；否则返回
  `prefill_port_table[target_dp_group][engine_rank]` 的 `{rank_ip, rank_port}`
  （`:197`-205）。
- `_ingest_put_extra` / `_extra_parallel_info`（`:207` / `:210`）：默认 no-op 的角色钩子。

KV 专属子类 `MooncakeKVBootstrapServer`（`mooncake/conn.py:93`）挂上 decode 需要的
prefill 侧布局字段：`_ingest_put_extra`（`:109`）记 `enable_mla_l1_5_cache`、
`kv_item_lens`/`kv_unit_lens`、`state_item_lens`/`state_unit_lens`（`:143`-151），处理
FlatKV vs legacy（拒绝混用 `:112`-115，校验 FlatKV peer layout `:116`-141）；
`_extra_parallel_info`（`:153`）把这些（FlatKV 时加 `flat_layout`）塞进 GET 同步响应。

**哪些元数据走哪条通道**：
- HTTP bootstrap：prefill 每个 `(dp_group, tp_rank)` 的 `(rank_ip, rank_port)`、
  prefill TP/DP size、MLA-L1.5 flag、KV/state item&unit 长度、可选 FlatKV layout。
  prefill 经 `_register_to_bootstrap`（`mooncake/prefill.py:1276`）PUT，decode 经
  `_get_prefill_parallel_info_from_server`（`mooncake/receiver.py:63`）+
  `_get_bootstrap_info_from_server`（`:110`）GET。
- ZMQ（稍后）：KV buffer **指针**、room id、decode 的 ZMQ `rank_port`、Mooncake
  `session_id`、dst 页索引。**buffer 指针不进 bootstrap 服务器**，是 decode 经 ZMQ 直
  接发给对应 prefill rank 的。

## 7. Mooncake 后端：数据面传输

### 7.1 类层次与状态 FSM

- `DisaggManagerBase`（`base/manager.py:31`）：传输中立、角色中立的基类——持
  `engine`、`server_socket = zmq.Context().socket(zmq.PULL)`（`:51`）、room 键控的状态
  FSM `request_status: dict[room -> status]`（`:55`）、`failure_records`（`:56`）。
  `update_status`（`:82`）单调递增，`Failed` **粘滞**（`:89`-93，晚到的 success 不能复活
  已坏的传输）；senders/receivers 只经 `check_status`（`:72`，未知抛 `KeyError`）或
  `room_status`（`:76`，未知返 `None`）读。
- `MooncakeKVManagerBase`（`mooncake/conn.py:35`）：加 KV 专属 args / MLA flag /
  dp-attention 接线 / metrics（`:45`-70），**在这里建 `MooncakeTransferEngine`**并注入
  （`:75`-79），`register_buffer_to_engine`（`:82`）把所有 `kv_data_ptrs` +
  `state_data_ptrs` 注册给引擎。
- `MooncakeKVManagerPrefill`（`mooncake/prefill.py:72`）/ `MooncakeKVManagerDecode`
  （`mooncake/decode.py:78`）：分别是 prefill / decode 的长期管理器（每引擎一个）。

**状态常量**在 `base/status.py:22` 的 `TransferPoll`（普通 int 常量类，非 Enum，值单调
好让 `update_status` 取 `max`）：`Failed=0`（`:23`，终态、粘滞）、`Bootstrapping=1`
（`:24`）、`Bootstrapped=2`（`:25`）、`WaitingForInput=3`（`:26`）、`Transferring=4`
（`:27`）、`Success=5`（`:28`）。`poll_and_all_reduce`（`utils.py:88`）把各 poller 的状
态做跨 gloo-group MIN all-reduce（`:104`-105），保证 TP 内各 rank 对同一请求的状态判断
一致。

### 7.2 Prefill 管理器（收预分配 + 发 KV）

`MooncakeKVManagerPrefill.__init__`（`prefill.py:73`）：建
`transfer_infos: dict[room][session] -> TransferInfo`（`:80`）、
`decode_kv_args_table: dict[session] -> KVArgsRegisterInfo`（`:81`），起 ZMQ 收线程
`start_prefill_thread()`（`:82`），注册 bootstrap（`:83`），起传输 worker 线程池
（`:125`）。

- **`start_prefill_thread`**（`:1127`）：`server_socket` 在新 `rank_port` 上 bind
  （`:1128`-1129）；内层 `bootstrap_thread`（`:1131`）循环 `recv_multipart`（`:1135`）：
  `room == "None"` 是 decode 的**指针注册** → `decode_kv_args_table[session] =
  KVArgsRegisterInfo.from_zmq(...)`（`:1143`-1146）；否则是**预分配** →
  `transfer_infos[room][session] = TransferInfo.from_zmq(...)`（`:1160`-1161），凑齐
  `required_dst_info_num` 个就置 `Bootstrapped`（`:1175`-1176）。
- **`add_transfer_request`**（`:1197`）：把一个 `TransferKVChunk` 塞进按
  `sum(session ports) % num_queues` 分片的传输队列（`:1232`-1255）；已失败/dummy room 早
  退（`:1217`-1230）。
- **`transfer_worker`**（`:915`）：核心发送循环。对 room 里每个 `TransferInfo`：跳
  dummy、查失败 session、`resolve_transfer_indices`（`:958`）算 src→dst 页映射，调
  `send_kvcache`（`:984`，普通）或 `send_kvcache_layerwise`（`:1000`，层流水），最后一
  个 chunk 收齐 `required_dst_info_num` 个 poll 后算 Success/Failed，调
  `sync_status_to_decode_endpoint` **带上 bootstrap_token**（`:1067`-1102）。
- **`send_kvcache`**（`:424`）：`group_concurrent_contiguous`（`utils.py:416`）把连续页归
  组，`_layer_transfer_blocks`（`:476`）/ `_flat_transfer_blocks`（`:451`）建
  `(src_addr, dst_addr, length)` 块，`_transfer_data` → `engine.batch_transfer_sync`
  （`:415`-422）。`send_kvcache_layerwise`（`:769`）逐层等 cache step
  （`_wait_until_cache_step` `:800`）边算边发。`send_mamba_cache`（`:635`）传 SSM 状态
  （`state_type=="mamba"` 时）。
- **`resolve_transfer_indices`**（`:252`）：处理 MLA-L1.5 的 4 种 OFF/ON 归属组合
  （`:282`-350）的 prefill→decode 索引映射。
- **`sync_status_to_decode_endpoint`**（`:853`）：**回推状态 + 首 token**——多帧
  `[room, status, prefill_rank, bootstrap_token, spec_candidate_payload]` 经 PUSH socket
  发给 decode（`:870`-878）。**首 token 走 ZMQ 不走 RDMA**（`entities.py:93` 注释确认）。
- **`abort_room`**（`:880`）：向该 room 每个预分配的 decode endpoint 推 `Failed`，避免
  decode 永久挂起。

**每请求发送句柄** `MooncakeKVSender`（`sender.py:35`）：`__init__` 置
`Bootstrapping`（`:45`）；`send(...)`（`:61`）对非末/末 chunk 分别调
`add_transfer_request`（`:92` / `:102`，末 chunk 带 `aux_index`/`bootstrap_token`/
`spec_candidate_ids`）；`send_layerwise`（`:115`）；`poll()`（`:161`）读
`kv_mgr.check_status(room)` 并套 bootstrap 超时（`:170`-181）。

### 7.3 Decode 管理器（注册指针 + 收首 token）

`MooncakeKVManagerDecode.__init__`（`decode.py:79`）：起 `start_decode_thread()`
（`:100`）、连接池、`required_prefill_response_num_table`、`prefill_response_tracker`、
`prefill_parallel_info`（`:101`-105）。

- **`start_decode_thread`**（`:112`）：`server_socket` 在新 `rank_port` bind
  （`:113`-114），建 `bootstrap_token_table`（`:118`）/ `spec_candidate_ids_table` 等；
  内层 `decode_thread`（`:123`）循环 `recv_multipart` → `parse_prefill_status_message`
  （`:59`）→ `_handle_prefill_status`（`:185`）；内层 `heartbeat_checker`（`:128`）轮询
  各 prefill `/health`，连续失败超阈调 `_handle_node_failure`（`:252`）。
- **`_handle_prefill_status`**（`:185`）：`Success` 时把 `prefill_rank` 加进
  `prefill_response_tracker[room]`（`:197`），token/spec 先进 `_pending_*` 表
  （`:198`-205），只有凑齐 `expected_response_num`（`:207`-212）才把 pending 提升进
  `bootstrap_token_table`/`spec_candidate_ids_table`（`:218`-225）并置 `Success`
  （`:226`）；`Failed` 记失败（`:229`-235）。
- **`pop_bootstrap_token`**（`:239`）/ **`pop_prefill_metadata`**（`:243`）：decode
  executor 的 `generate_events` 取首 token / spec 用（见 §3）。
- **`_handle_node_failure`**（`:252`）：踢掉死 prefill 的连接，把受影响 room 标
  `Failed`。

**每请求接收句柄** `MooncakeKVReceiver`（`receiver.py:469`，类级缓存 PUSH socket
`_socket_cache` `:471`）：`__init__` 置 `Bootstrapping`（`:488`），取 prefill 并行信息
（`:496`），算路由计划 `_calc(...)`（`:504`，见 §9），
`target_dp_group = room % dp_size`（`:513`），`_register_kv_args()`（`:530`）把本地
buffer 指针发给 prefill，末置 `Bootstrapped`（`:535`）。
- **`_register_kv_args`**（`:601`）：发 `room=b"None"` 的多帧，带 **decode 的 buffer 指
  针**：`[b"None", ip, rank_port, session_id, packed_kv_ptrs, b"", packed_state_ptrs,
  decode_prefix_len]`（`:621`-633）——填 prefill 的 `decode_kv_args_table`。
- **`prefill(kv_indices, aux_index, decode_prefix_len, ...)`**（`:645`）：发**预分配**多
  帧（`:698`-747），`is_dummy` 把数据帧留空。`poll()`（`:750`）读状态并在
  `WaitingForInput` 套 `waiting_timeout`（`:755`-769）。

### 7.4 交换的 wire 格式（entities.py）

- **`TransferKVChunk`**（`entities.py:85`）：prefill 侧的一个发送工作项——`room`、
  `prefill_kv_indices`、`index_slice`、`is_last`、`prefill_aux_index`、`mla_l1_5_args`、
  `prefill_mamba_indices`、`bootstrap_token=-1`（`:94`）、layerwise 相关、
  `spec_candidate_ids`、`flat_manifest`。
- **`TransferInfo`**（`entities.py:110`）：decode 的预分配（prefill 侧看到的）——`room`、
  `endpoint`、`dst_port`、`mooncake_session_id`、`dst_kv_indices`、`dst_aux_index`、
  `required_dst_info_num`、`decode_prefix_len`、MLA-L1.5 mask 三件套、`dst_mamba_indices`、
  `is_dummy`、`transfer_fragments`、`flat_manifest`、`flat_peer_layout`。
  `from_zmq`（`:130`）是预分配的**权威 wire 解码**：`is_dummy` iff
  `msg[4]==b"" and msg[5]==b""`（`:134`），后续帧解析索引/mask/fragments/FlatKV
  （`:145`-199）。
- **`KVArgsRegisterInfo`**（`entities.py:204`）：decode 的指针注册——`room`、`endpoint`、
  `dst_port`、`session_id`、`dst_kv_ptrs`、`dst_aux_ptrs`、`dst_state_data_ptrs`、
  `decode_prefix_len`；`from_zmq`（`:215`）用 `struct.unpack("<N>Q", ...)` 解指针数组
  （`:233`-237），兼容旧 sender 省掉 state 帧的情况（`:218`-226）。
- **`KVTransferError`**（`entities.py:67`）：带 `bootstrap_room`/`failure_reason`/
  `remote_endpoint`。

## 8. FlatKV PD 契约

flat-state / LCM two-level 布局（见 `kvcache-management.md` 的 LCM 章节）下，KV 不再是
「逐层连续 buffer + 页索引向量」，而是多个 cache group 共享一个页 ID 命名空间、绑到别
名化的原始 slab。prefill / decode 各自独立分配页，所以两边要交换**结构化、带版本、可
交叉校验**的 layout + manifest。这套契约（model-neutral）在 `flatkv.py`，协议版本
`FLATKV_PD_PROTOCOL_VERSION = 1`（`:37`）。

关键类型：
- **`FlatKVPDLayout`**（`:345`）：一侧本地的完整 flat cache ABI + 容量——`version`、
  `layout_fingerprint`（64 位 SHA-256 hex）、`block_size`、`num_pages_with_null`、
  `physical_buffer_ids`、`physical_page_bytes`、`groups`。`__post_init__`（`:357`）做大
  量不变量校验（slot 覆盖、segment 不越界、无重复 group）。
- **`FlatKVPDPeerLayout`**（`:302`）：peer 只需的子集（`version` +
  `layout_fingerprint` + `num_pages_with_null`），跨节点比对 ABI 用；
  `to_wire_bytes`/`from_wire_bytes`（`:318` / `:325`）走 canonical JSON。
- **`FlatKVPDGroup`**（`:223`）：一个 cache group——`group_id`、`family`
  （`history`/`state`）、`transfer_policy`（`full_suffix`/`latest_snapshot`）、
  `physical_slots`、`cache_blocks_per_lcm_block`、`transfer_segments`（每个是
  `FlatKVPDTransferSegment` `:199`，描述一个 strided 字段拷贝）。
- **`FlatKVPDPageManifest`**（`:457`）：一个请求的逐 group 选页——`groups`（每个
  `FlatKVPDGroupPages` `:438`：`group_id` + `page_ids`）+ `prefix_len` + `prompt_len` +
  `version`。同样 `to_wire_bytes`/`from_wire_bytes`（`:487` / `:494`）。

关键函数：
- **`build_flatkv_page_manifest(forward_op, layout, request_row, prefix_len,
  prompt_len)`**（`:671`）：从 scheduler 的 `flat_block_tables_arrays()` 按各 group 的
  `transfer_policy` 选页（`full_suffix` 选 `[prefix, prompt)` 全后缀，
  `latest_snapshot` 只选最后一块，`_logical_slots` `:545`），并校验没有别名 live 未选
  slot（`:773`-787）。prefill 的 `_flat_decode`、decode 的 `_flat_prefill` 都用它。
- **`validate_flatkv_manifest_pair(src, dst, layout, dst_num_pages_with_null)`**
  （`:621`）：收发双方 manifest 交叉校验——group 顺序、页对齐、页数符合 policy、
  prefix_len/prompt_len 一致（`:640`-646）。
- **`validate_flatkv_slab_registrations(...)`**（`:791`）：校验原始 slab 注册与 layout
  一致（数量、slot 顺序、buffer_id、extent、无重叠）。
- **`build_lcm_flatkv_pd_contract(plan, backing, group_specs, field_dtypes)`**
  （`:844`）：从一个 LCM arena 造 `(FlatKVPDLayout, registrations)`——不拷贝不摊平，直接
  按 arena 的整数几何算 `transfer_segments` 的 `page_zero_offset`/`page_stride_bytes`
  （`:886`-903），并对整个 plan 算 SHA-256 `layout_fingerprint`（`:915`-963）。

契约的产出方是 KV pool：`LcmCachePool.pd_contract(group_specs)`
（`layers/attention/kv_cache/lcm.py:80`）内部调 `build_lcm_flatkv_pd_contract`
（`:94`）；`LcmMHATokenToKVPool.get_flatkv_pd_contract`
（`kv_cache/lcm_mha.py:212`）/ `LcmMLATokenToKVPool.get_flatkv_pd_contract`
（`kv_cache/lcm_mla.py:183`）是 `get_kv_args` 走的 ABI 入口。

## 9. 异构 TP 的传输分片规划器（transfer_plan.py）

当 prefill 和 decode 的 TP size 不同（或 GQA/MQA 下 KV head 数少于 TP 数），一个 decode
rank 需要的某个 KV 分片可能散落在多个 prefill rank 上，反之亦然。`PDTransferPlanner`
（`transfer_plan.py:204`）为一个 decode rank 算出该从哪些 prefill rank、拷哪些字节区
间。

- 输入：`ParallelLayout`（`:41`，role/world_size/dp_size）、每个 buffer 的
  `BufferLayout`（`:62`，`logical_axis`（kv_head/state_channel/replicated）/
  `logical_size`/`item_stride_bytes`/`tp_replica_group_size` 等）。
- **`plan_for_decode_rank(decode_rank)`**（`:221`）：TP 相等且无 replica 分组时走
  **identity plan**（`:229`-246，一对一，`plan_kind="identity"`），否则对每个 buffer
  按逻辑区间求交（`_rank_interval_for_buffer` `:450`，`intersect` `:196`）算出
  `TransferFragment`（`:93`，src/dst rank + 字节偏移 + 每页字节数），产出
  `plan_kind="fragmented"` 的 `RankTransferPlan`（`:174`）。
- `_calc_source_fanout`（`:387`）预算每个 prefill rank 要响应多少 decode rank
  （`required_dst_info_num`，即 §7.2 里凑齐才 `Bootstrapped` 的那个数）。
- `TransferFragment` 经 `encode_transfer_fragments`/`decode_transfer_fragments`
  （`:110` / `:134`，`TRANSFER_PLAN_PROTOCOL_VERSION=1` `:107`）序列化进预分配 ZMQ 消息
  的第 12/13 帧（`entities.py:168`-170）。

`MooncakeKVReceiver._calc`（`receiver.py:378`）是调用点：FlatKV（`:384`）、MLA-L1.5
（`:417`）、非-MLA 经 `PDTransferPlanner`（`:429`-430）、MLA 等/大/小 TP 分支
（`:432`-466），产出 `ReceiverRoutePlan`（`:131`）。

## 10. MOONCAKE_ASYNC 异步后端现状

`MooncakeAsyncKVManager`（`async_conn.py:70`）是走 Mooncake 异步 submit/poll API
（`transfer_submit_write` → `batch_id` → `transfer_check_status`）而非同步 batch write
的另一套 prefill 管理器，核心是层流水状态机 `async_transfer_worker`（`:265`，内含
`discard_finished_bid_inplace` `:266` 解码状态码 `1`=done/`-1`=timeout/`-2`=failed、
`submit_transfer` `:374`、`pop_transferred` `:458`、`finalize` `:504`）。

> ⚠️ **当前该路径与重构后的同步路径不同步，构造即失败。**
> `MooncakeAsyncKVManager.__init__`（`:71`-81）签名是
> `(args, disaggregation_mode, server_args, is_mla_backend, draft_is_mla_backend)`，并
> `super().__init__(...)` 用同样 5 参调父类；但它的父类经 `import ... as
> MooncakeKVManager`（`:37`）其实是 `MooncakeKVManagerPrefill`，后者当前签名只有
> `(args: KVManagerArgs, kv_args: KVArgs)`（`prefill.py:73`）。它还引用
> `args.offsets`（`:92`）、`server_args.kv_cache_quant_method`（`:87`），都对不上现在
> 的 `KVArgs`/`KVManagerArgs`。这说明 `MOONCAKE_ASYNC` 是**遗留/未维护**分支，不要当成
> 和同步 `MOONCAKE` 对等的可用后端。新代码走 `MOONCAKE`。

## 关键数据结构

| 数据结构 | 位置 | 作用 |
| --- | --- | --- |
| `DisaggPrefillExecutor` / `DisaggDecodeExecutor` | `prefill_executor.py:48` / `decode_executor.py:44` | PD 两侧每步编排 + C++ 事件泵；持每请求 sender/receiver。 |
| `MooncakeKVManagerPrefill` / `MooncakeKVManagerDecode` | `mooncake/prefill.py:72` / `decode.py:78` | 每引擎一个的长期管理器：ZMQ 收线程 + 传输 worker / 心跳。 |
| `MooncakeKVSender` / `MooncakeKVReceiver` | `mooncake/sender.py:35` / `receiver.py:469` | 每请求一个的发送/接收句柄。 |
| `MooncakeTransferEngine` | `base/mooncake_engine.py:26` | 包 `mooncake.engine.TransferEngine`，单边 RDMA 写。 |
| `DisaggBootstrapServerBase` / `MooncakeKVBootstrapServer` | `base/bootstrap.py:42` / `mooncake/conn.py:93` | HTTP rendezvous（prefill PUT ip:port，decode GET）。 |
| `DisaggManagerBase` | `base/manager.py:31` | engine + ZMQ 控制 socket + room 键控状态 FSM。 |
| `TransferPoll` | `base/status.py:22` | 6 个状态常量（Failed 粘滞、单调递增）。 |
| `KVArgs` / `KVManagerArgs` | `mooncake/entities.py:39` / `:242` | buffer 描述符 / 引擎级配置。 |
| `TransferInfo` / `KVArgsRegisterInfo` / `TransferKVChunk` | `mooncake/entities.py:110` / `:204` / `:85` | 预分配 / 指针注册 / 发送工作项的 wire 结构。 |
| `PD.{Bootstrapped,Failed,Succeeded,RemotePrefillDone}Event` | `tokenspeed-scheduler/csrc/scheduler/outside_events/pd.h:30`-`:54` | C++ PD 事件 variant。 |
| `FlatKVPDLayout` / `FlatKVPDPageManifest` | `flatkv.py:345` / `:457` | FlatKV 的结构化 layout / 逐请求选页 manifest。 |
| `PDTransferPlanner` / `TransferFragment` | `transfer_plan.py:204` / `:93` | 异构 TP 的传输分片规划。 |
| `BootstrapInfo` | `base/bootstrap.py:35` | 逐请求 rendezvous 三元组（host/port/room）。 |
| `MetadataBuffers` | `utils.py:288` | 首 token metadata 的 RDMA buffer（output_ids/logprobs）。 |
| `StepCounter` | `utils.py:433` | layerwise 传输的 cache/aux step 计数（CUDA event fence）。 |

## 读代码建议

想快速掌握 PD 主线，按此顺序读：

1. `python/tokenspeed/runtime/utils/server_args.py:635`（`resolve_disaggregation`：三种
   模式各自约束）
2. `python/tokenspeed/runtime/pd/factory.py:188`（`create_kv_transfer` +
   `get_kv_args`：造 Executor 与 buffer 描述符）
3. `python/tokenspeed/runtime/engine/event_loop.py:886`（`_dispatch_forward` 四条
   path——PD 的编排主干）
4. `python/tokenspeed/runtime/pd/decode_executor.py:159` 和
   `prefill_executor.py:244`（一次预分配 / 一次发 KV 的具体索引计算）
5. `python/tokenspeed/runtime/pd/mooncake/prefill.py:915`（`transfer_worker`：真正搬 KV
   的循环）+ `mooncake/receiver.py:601`（decode 注册指针）
6. `tokenspeed-scheduler/csrc/fsm/pd_events.cpp:40`（首 token 如何注入 decode 请求的
   FSM）

想深入特定子系统：

- **控制面 rendezvous / ZMQ 协议**：`base/bootstrap.py`、`mooncake/conn.py`、
  `mooncake/entities.py` 的 `from_zmq`。
- **数据面 RDMA**：`base/mooncake_engine.py`、`mooncake/prefill.py`（`send_kvcache` /
  `send_kvcache_layerwise`）、`mooncake/decode.py`（`_handle_prefill_status`）。
- **FlatKV / LCM PD**：`flatkv.py`、`layers/attention/kv_cache/lcm.py:80`
  （`pd_contract`）、`lcm_mha.py` / `lcm_mla.py` 的 `get_flatkv_pd_contract`。
- **异构 TP**：`transfer_plan.py`、`mooncake/receiver.py:378`（`_calc`）。
- **C++ 事件与 FSM**：`tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`、
  `fsm/pd_events.cpp` / `pd_events.h`、`scheduler/outside_events/pd.h`。
