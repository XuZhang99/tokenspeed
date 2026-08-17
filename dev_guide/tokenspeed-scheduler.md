# tokenspeed-scheduler 开发指南

`tokenspeed-scheduler/` 是 TokenSpeed 的 C++ 请求状态机、cache coordinator 与调度
核心，通过 nanobind 暴露给 Python event loop。当前只有一条 cache 路径；旧的
`TOKENSPEED_FLAT_KVCACHE`、Python radix tree 和 `PagedCache*` 类型名均已移除。

## 总览

```text
Python event_loop
  ├─ submit_requests(RequestSpec[])
  ├─ next_execution_plan()
  └─ advance(ExecutionEvent)
           │
           ▼
C++ Scheduler
  ├─ Request + FSM
  ├─ CacheCoordinator
  ├─ ReqPoolAllocator
  ├─ TierTransferManager
  └─ ExecutionPlan
       ├─ Forward.Batch
       ├─ Cache.LoadBackOp / WriteBackOp
       └─ pages_to_zero[group_id]
```

## 1. 目录结构

```text
tokenspeed-scheduler/
├── bindings/python_module.cpp
├── csrc/
│   ├── cache/
│   │   ├── core/          # config, block pool/table/ref, acquire plan
│   │   ├── prefix/        # hashing, index, matcher
│   │   ├── allocator/     # token-free GroupAllocator
│   │   ├── coordinator/   # GroupGeometry, admission, facade
│   │   └── tier/          # Host/Device transfer manager
│   ├── fsm/               # forward 与 PD states/events
│   ├── resource/allocator/# request-pool index
│   └── scheduler/
│       ├── scheduler.*
│       ├── request.*
│       ├── operations/    # forward/cache plan builder
│       ├── outside_events/
│       ├── execution_plan.h
│       └── kv_cache_events.*
├── python/tests/
└── tests/cpp/
```

## 2. Python API

[`bindings/python_module.cpp`](../tokenspeed-scheduler/bindings/python_module.cpp)
暴露单一 module `tokenspeed_scheduler_ext`，Python package 再重导出。

### 2.1 配置

`SchedulerConfig` 的关键字段：

- `prefix_granularity`；
- `num_device_pages` / `num_host_pages`；
- `cache_groups: list[CacheGroupConfig]`；
- `max_scheduled_tokens`、`max_batch_size`；
- `decode_input_tokens`、`overlap_schedule_depth`；
- `role: P | D | Fused`、`enable_pd_cache`；
- `disable_l2_cache`、`enable_l3_storage`、`enable_kv_cache_events`；
- `enable_mixed_prefill_decode`、`disable_prefix_cache`、`prefix_replay_tokens`。

`CacheGroupConfig` 是 Python/C++ 边界类型：

```text
group_id
rows_per_page
entry_stride_tokens
total_pages
cache_blocks_per_lcm_block
retention
sliding_window_tokens
family
transfer_policy
```

enum 名为 `CacheRetention`、`CacheGroupFamily` 和 `CacheTransferPolicy`。snapshot state
在 Python bridge 处折成 rows/stride；C++ 内统一通过 `BlockGranularity()` 读取 slot
token span。

`SchedulerConfig::Validate()` 是唯一配置 gate，在成员构造前验证：

- 所有 scalar 的范围；
- 每个 group 自身合法性；
- group block span 正向整除 P；
- PD transfer policy；
- recurrent-state chunk alignment 等跨字段约束。

`MakeSpecsFromConfig()` 只做翻译，不重复校验。

### 2.2 请求与执行

- `RequestSpec`：`request_id`、`tokens`、`max_new_tokens`；
- `Scheduler.submit_requests()`；
- `Scheduler.next_execution_plan()`；
- `Scheduler.advance(ExecutionEvent)`；
- `Scheduler.drain_kv_events()`；
- cache control/metrics：`clear_l1_cache()`、`clear_cache()`、
  `available_kv_pages()`、`active_kv_pages()`、每组 total/available blocks 等。

`ExecutionPlan` 的 nanobind view 将内部 operation variant 分为 forward 与 cache 两类，
并单独暴露 `pages_to_zero`。

## 3. Request FSM

`csrc/fsm/` 把 forward 与 PD 生命周期建模成 typed states/events。主要 forward state：

```text
Submitted
  → Prefilling
  → PrefillDone
  → Decoding
  → Finished

Decoding/PrefillDone → Retracted → Submitted/Finished
```

PD 还包含 bootstrapping、remote-prefill completion 与 success/failure transition。
`Request` 持有 token container、forward state、PD state、per-group block tables 与 cache
progress；非法 event/state 组合在 C++ 侧显式失败。

外部事件由 `outside_event_handler.cpp` 分派：

- `ForwardEvent.ExtendResult / Finish / Abort / UpdateReserveNumTokens`；
- `Cache.WriteBackDoneEvent / LoadBackDoneEvent`；
- `PD.BootstrappedEvent / RemotePrefillDoneEvent / SucceededEvent / FailedEvent`。

## 4. `NextExecutionPlan()`

调度入口位于 `scheduler.cpp`，forward 构建主体在
`scheduler/operations/forward.cpp::buildForwardOperations()`。

一个 iteration 的主要阶段：

1. 选择 waiting/prefilling/decoding candidate；
2. 对新请求 probe prefix，并构造 per-group token demand；
3. 通过 coordinator admission；失败时评估 cache eviction/retraction；
4. 安排 first prefill、continuation prefill 或 decode；
5. 生成 generic group block tables、input ids、position/length 元数据；
6. 合并 Host tier load/writeback operation；
7. 收集新获得的 child blocks 到 `pages_to_zero`；
8. 返回 immutable plan，等待 Python 执行并回灌 event。

`PlanBuildContext` 记录 admission failure、store ACK barrier 与 capacity blocker，避免
在多处用隐式布尔值改变计划。

## 5. Prefill 与 decode

### Prefill

`schedulePrefillFirstChunk()` 先完成 prefix probe/admission，随后按 token budget、state
alignment 和 context limit 切 chunk。`schedulePrefill()` 继续已有请求。chunk 完成后，
Python 用 `ExtendResult` 回灌实际 token；scheduler 发布完成的 prefix boundary。

### Decode

`scheduleDecode()` 为每个 active request 预留 `decode_input_tokens`，overlap 模式还要
覆盖下一轮未 commit 的 horizon。Forward result 回灌后才把 accepted token 与 cache
progress 提交到 request state。

### Retraction

容量不足时，scheduler 根据 coordinator 给出的可释放 parent、Host tier 能力与请求
状态选择 victim。能写回 Host 时先生成 `WriteBackOp`，完成 event 后进入 recovery；
否则释放并重新排队。Decode role 的 Host cache是 best-effort recovery，不是持续镜像。

## 6. ExecutionPlan operation

### `Forward.Batch`

主要携带：

- request ids、request-pool indices；
- input/shifted/decode ids；
- prefill lengths、extend-prefix lengths；
- 每组 block table 数组和 base offset；
- local/remote prefill 等 mode 信息。

Python 将 generic block table 转成 backend 所需 page/state table，再填 executor input
buffers。

### Cache operation

- `Cache.LoadBackOp`：Host → Device；
- `Cache.WriteBackOp`：Device → Host；
- 字段包含 op ids、group ids、source/destination block ids。

### `pages_to_zero`

这是 `map<group_id, child block ids>`。LCM parent 可能还有 live sibling child，所以
必须保留 group identity，只清零该 group 的 field payload，不能对 parent 做无差别
全量清零。

## 7. Cache architecture

详细设计见 [KV Cache 管理机制](kvcache-management.md)。边界如下：

```text
CacheGroup
  = logical spec
  + PrefixMatcher / PrefixCacheIndex
  + GroupAllocator

CacheCoordinator
  = 所有 group 的 request-level facade
```

- prefix 目录感知 token identity 与 reuse policy；
- allocator 目录只感知 physical blocks；
- `GroupGeometry` 负责 token → block count；
- 多个 group 共享 Device/Host `BlockPool`；
- scheduler 只通过 coordinator probe/admit/publish/free。

调度层允许 transport `cache_blocks_per_lcm_block`，但物理 capacity 算术必须留在 cache
层。标识符 `page_size` 不应出现在 `tokenspeed-scheduler`：scheduler 的通用术语是
`block_granularity`。

## 8. Prefix cache 与 KV events

`PrefixCacheIndex` 维护 pool-scoped CacheKey → block；full attention prefix-closed，SWA
使用 resumable suffix boundary。Device/Host tier 可共享同一逻辑 index 语义。

启用 KV events 时，scheduler 对完成 boundary 建 block hash，cache mutation sink 再
产出 `BlockStored` / `BlockRemoved`。该 wire hash 实现在 `kv_cache_events.cpp`，与内部
prefix SHA-256 hash 是独立格式，不应无意混用。

## 9. 构建与测试

子项目入口：

```bash
cd tokenspeed-scheduler
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Python binding 测试位于 `tokenspeed-scheduler/python/tests/`。仓库也提供根级 setup
命令与 CI workflow；实际执行应使用项目当前环境和对应平台容器。

## 10. 维护检查表

- 新配置字段必须加入 C++ `Validate()`、nanobind 和 Python bridge 测试；
- scheduling/FSM 不做 LCM packing 算术；
- allocator 不读取 token 粒度；
- state snapshot 不使用 page 命名；
- 新 operation 必须有 plan 输出、Python 消费和 completion event 的完整闭环；
- 修改 block table 语义时同时验证 eager、CUDA graph、draft 和 PD 路径；
- commit 前按仓库要求运行 `pre-commit run --all-files`。
