# tokenspeed-scheduler 目录详解
##### commit-id: d50bb481505c229f93594e56d5e1f8a772876451

本文详细介绍 `tokenspeed-scheduler/` 这个子项目：它是 TokenSpeed 的**核心调度器**，
一个用 C++20 写、通过 nanobind 暴露给 Python 的独立扩展模块。Python 侧的
`event_loop.py` 每个迭代都调用它来决定"这一步 forward 跑哪些请求、每个请求占哪些
KV 页、要不要做 prefix 复用 / 换入换出 / 抢占"。它是 `inference-flow.md`
（一次请求端到端流程）里 "scheduler subprocess: RequestHandler + C++ Scheduler"
那一环的展开，也是 `kvcache-management.md` 里"分配决策（C++ scheduler）"层的完整实现。

> 约定：全文用大量 `path:line` 锚点，路径相对 `tokenspeed-scheduler/`。行号会随无关
> 改动漂移，更新本文时请逐个用 `grep -n` / `sed -n` 核对。

## 总览

调度器只做**决策**，不持有 GPU 显存。它维护每个请求的状态机（FSM），在每个迭代里
挑一批请求、算好它们要占的逻辑页，产出一份 `ExecutionPlan` 交给 Python runtime 去真正
跑 forward、搬 KV。

```text
Python event_loop.py（每个迭代）
  │
  │ submit_requests(specs)        ← 新请求进入（Submitted 状态）
  ▼
┌─ C++ Scheduler ────────────────────────────────────────────────────┐
│  requests_: id → Request（每个请求一个 FSM）                         │
│                                                                      │
│  next_execution_plan():                                             │
│    1. 回收 Finished，生成 Draining 请求的 WriteBack op               │
│    2. newForwardOperation(candidates):                              │
│         按优先级排序 → 逐个请求过"调度门"(schedulePrefill/Decode…)  │
│         每个门：prefix 匹配 + 容量检查 + 驱逐 → 通过则 Apply 事件    │
│         推进 FSM 并分配页，产出 Prefill/Decode Operation            │
│    3. 打包成 ExecutionPlan（forward ops + cache ops + 要清零的页）   │
│  ▲                                                                   │
│  │ advance(ExecutionEvent)      ← forward/PD/cache 完成事件回灌      │
│  │   ExtendResult / Finish / Abort / WriteBackDone / LoadBackDone…  │
└──┼──────────────────────────────────────────────────────────────────┘
   │ next_execution_plan() 返回
   ▼
Python runtime：清零新页 → 提交 cache op（H2D/D2H）→ 跑 forward → 采样
                → 把结果 advance() 回灌调度器
```

两条互斥的 KV 后端由**编译期**开关 `TOKENSPEED_FLAT_KVCACHE`（`CMakeLists.txt:12`）选择：

- **radix 路径（默认，flag=OFF）**：`csrc/resource/` 下的 radix tree + `LocalKVAllocator`
  + `PageAllocator`，prefix 缓存靠 `KVPrefixCache` / `HybridPrefixCache`。
- **flat 路径（flag=ON）**：`csrc/cache/` 下的 `KvCacheCoordinator` + `BlockPool`，
  面向 flat-state / LCM two-level 布局（见 `kvcache-management.md` 的 LCM 章节）。

`scheduler.h`、`scheduler.cpp`、`operations/forward.cpp`、`outside_event_handler.cpp`
都用 `#if [!]TOKENSPEED_FLAT_KVCACHE` 双开。Python 侧靠 `FLAT_KVCACHE` 常量
（`__init__.py:53`，来自 `python_module.cpp:116-120`）判断当前扩展是哪种 build。

## 目录结构

```text
tokenspeed-scheduler/
├── CMakeLists.txt          构建（scikit-build-core + nanobind + spdlog + OpenSSL）
├── pyproject.toml          包名 tokenspeed-scheduler，wheel 打 python/tokenspeed_scheduler
├── bindings/
│   └── python_module.cpp   nanobind 绑定：唯一的 Python↔C++ 边界
├── csrc/
│   ├── core/               token_container：请求 token 序列 + 分页视图
│   ├── scheduler/          调度主逻辑：Scheduler、Request、ExecutionPlan、operations/
│   ├── fsm/                请求状态机：states / *_events / *_states（forward/cache/pd）
│   ├── resource/           radix 路径资源层
│   │   ├── allocator/      PageAllocator / OwnedPages / LocalKVAllocator /
│   │   │                   ReqPoolAllocator / PagedCacheGroup* / Mamba*Allocator
│   │   ├── radix_tree/     RadixTree / TreeNode（prefix 树）
│   │   ├── kv_prefix_cache/    KVPrefixCache（KV-only 两级 prefix 缓存）
│   │   └── hybrid_prefix_cache/ HybridPrefixCache（KV + Mamba 状态 + paged-cache）
│   └── cache/              flat 路径：KvCacheCoordinator / BlockPool / 各 attention Manager
├── python/
│   ├── tokenspeed_scheduler/__init__.py   Python 包（re-export 扩展符号）
│   └── tests/              10 个 pytest（cache 绑定 / FSM / paged-cache / PD / flat）
└── tests/cpp/              41 个 gtest（生命周期 / 驱逐 / 抢占 / prefetch / mamba…）
```

## 一、Python API 边界（`bindings/python_module.cpp`）

整个扩展只有一个 nanobind module `tokenspeed_scheduler_ext`（`python_module.cpp:108`），
`python/tokenspeed_scheduler/__init__.py` 只是把符号 re-export 出来。核心暴露：

- **`Scheduler`**（`python_module.cpp:477`）——唯一有状态的对象。方法映射：
  - `submit_requests(specs)` → `SubmitRequests`（`:479`）
  - `next_execution_plan()` → `NextExecutionPlan`（`:482`）
  - `advance(event)` → `Advance`（`:483`）
  - `drain_kv_events()`（`:484`）——排空 prefix-cache 的 BlockStored/BlockRemoved 事件
  - 一堆只读监控：`waiting_size` / `decoding_size` / `available_kv_pages` /
    `active_kv_pages` / `paged_cache_group_*`（`:494-513`）
- **`SchedulerConfig`**（`python_module.cpp:249`，结构体在 `scheduler/types.h:77`）——
  一次性传入的配置：`block_size`、`max_scheduled_tokens`、`max_batch_size`、
  `decode_input_tokens`、`role`（P/D/Fused）、`num_device_pages`/`num_host_pages`、
  `paged_cache_groups`、各种开关（`disable_prefix_cache` / `enable_mamba` /
  `enable_mixed_prefill_decode` / `enable_flatkv_pd` …）。
- **`RequestSpec`**（`python_module.cpp:277`，结构体 `scheduler/request_spec.h:30`）——
  提交一个请求需要的最小信息：`request_id` + `tokens` + 可选的 `rolling_hashes`
  （L3 storage 命中用）+ `storage_hit_pages`。
- **输入事件**（Python → 调度器，通过 `advance`）：
  - `ForwardEvent.{ExtendResult, Finish, Abort, UpdateReserveNumTokens}`（`:284-302`）
  - `Cache.{PrefetchDoneEvent, WriteBackDoneEvent, LoadBackDoneEvent}`（`:309-324`）
  - `PD.{BootstrappedEvent, FailedEvent, SucceededEvent, RemotePrefillDoneEvent}`（`:326-341`）
- **输出**（调度器 → Python）：`ExecutionPlan`（`:469`），暴露 `forward`（一批
  `Forward.FlatForwardOp`）、`cache`（`WriteBackOp`/`LoadBackOp`/`PrefetchOp`/`BackUpOp`）、
  `flat_pages_to_zero`、`flat_oom_request_ids`/`flat_terminal_errors`。

`ExecutionEvent`（`:343`）是事件容器，Python 用 `add_event` 往里塞多个事件，一次
`advance` 全部消费。绑定顶部注释（`python_module.cpp:39-49`）明确：只有 Config/
RequestSpec/事件是"可写"类型，其余都是调度器产出、Python 只读。

## 二、请求状态机（`csrc/fsm/`）

每个请求是一个 `Request`（`scheduler/request.h:51`），内部持有一个
`fsm::State`——13 个状态的 `std::variant`（`fsm/states.h:32`）：

```text
Bootstrapping  ─(PD only)→ Submitted
Submitted ─prefill首块→ Prefilling ─(还有块)→ Prefilling
                              └─(最后块)→ PrefillDone ─decode→ Decoding ─decode→ Decoding
Prefetching → PrefetchDone（host prefetch 完成，等同 Submitted 再调度）
任意 forward 态 ─Finish→ Draining ─(writeback 生成)→ WritingBack ─(完成)→ Finished
Decoding/PrefillDone ─抢占→ Retracting ─(writeback完成)→ Retracted ─(恢复)→ Decoding
任意态 ─Abort→ Aborting/Finished
```

设计要点：

- **状态即资源所有权**。状态对象直接持有该阶段独占的资源句柄（RAII）：`BaseState`
  （`fsm/forward_states.h:106`）持 `DeviceNodeRef`（radix 树节点 pin）+
  `LocalKVAllocator`（本地尾页）+ 可选 `LocalMambaAllocator`；`ForwardState`
  （`:174`）在此之上加 `ReqPoolIndex`（Python 请求池的行号，1-based）。状态转移时资源
  在事件里 `Take*` 移动到下一个状态，析构顺序即释放顺序。
- **事件 = 转移函数 + 资源分配**。事件定义在 `fsm/forward_events.h`（forward 类）、
  `cache_events.h`（prefetch）、`pd_events.h`（PD）。`Request::Apply`（`request.h:60`）
  用 `std::visit` 把当前状态喂给事件的 `operator()`，返回新状态。非法转移由
  `InvalidTransitionHandler` 兜底抛异常。
- 关键 forward 事件：`SchedulePrefillFirstChunkEvent`（`forward_events.h:67`，
  Submitted→Prefilling/PrefillDone，做 prefix 匹配 + 首块分配）、`SchedulePrefillEvent`
  （`:144`，续块）、`ScheduleDecodeEvent`（`:175`）、`ScheduleDecodeFromRetractedEvent`
  （`:204`，Retracted→Decoding 恢复，含 host loadback）、`FinishEvent`（`:240`，
  决定进 Draining 还是直接 Finished）、`AbortEvent`（`:286`）、`ScheduleRetractEvent`
  （`:332`，抢占）、`ExtendResultEvent`（`:402`，把 forward 出的新 token 追加进
  token 容器，并在跨页时把满页 publish 进 prefix 缓存）。
- flat 路径独有：`FlatRetractEvent`（`forward_events.h:316`）——flat 抢占是"释放所有
  页、整个请求重排成一次新 prefill"，prefix 靠 hash-intact 的页释放 + L2 恢复。

`StateName()`（`request.h:282`）给出可读状态名，调试和错误信息都用它。

## 三、调度主循环（`csrc/scheduler/`）

### Scheduler 类（`scheduler.h:61`，实现分散在 `scheduler.cpp` + `operations/`）

构造函数（`scheduler.cpp:57`）按 config 建好所有资源：`device_allocator_` /
`host_allocator_`（radix 的物理页池）、`kv_prefix_cache_`、`req_pool_allocator_`，
以及可选的 `hybrid_prefix_cache_`（当配了 mamba / paged-cache / prefix-adjunct 时，
`scheduler.cpp:120`）。flat build 额外建 `block_pool_` / `flat_host_pool_` /
`coordinator_`（`scheduler.cpp:64-77`）。

三个入口方法：

1. **`SubmitRequests`**（`scheduler.cpp:203`）——把每个 `RequestSpec` 包成 `Request`
   （初始 Submitted，PD decode 角色为 Bootstrapping）塞进 `requests_`。
2. **`NextExecutionPlan`**（`scheduler.cpp:367`）——一个迭代的核心：
   - 先给 Draining 请求生成 WriteBack op（`newWriteBackOperation`，`scheduler.cpp:338`），
     并回收 Finished 请求（从 `requests_` erase + 通知 prefix 缓存释放）。
   - 收集候选（排除 Draining/Prefetching/Retracting/WritingBack 这些"在途"态），
     调 `newForwardOperation` 得到本轮 forward ops + cache ops。
   - 打包进 `ExecutionPlan`：forward 合并成一个 `FlatForwardOperation`，writeback/
     loadback 各自成 `CacheOperation`，flat 路径还塞 `flat_pages_to_zero`（新分配、
     forward 前要清零的页）和 `flat_oom_request_ids`。
3. **`Advance`**（`scheduler.cpp:530`）——双层 `std::visit` 分发
   `ExecutionEvent` 里的每个事件到对应的 `handleEvent` 重载
   （实现全在 `outside_event_handler.cpp`）。

### 核心：newForwardOperation（`operations/forward.cpp:1095`）

这是"这一步跑哪些请求"的决策心脏：

1. **优先级排序**（`forward.cpp:1099`）：Prefilling(1) < Submitted(2) < Decode(3)；
   开了 mixed-batch 时 decode 提到 0。**同优先级按 `Id()` 排序**——保证 TP 各 rank
   调度完全一致的子集（rank 间不一致会让 NCCL 死锁，`forward.cpp:1109`）。
2. **token 预算循环**（`forward.cpp:1151`）：`token_budget = max_scheduled_tokens`，
   逐个候选过对应的"调度门"，通过就 `Apply` 事件推进 FSM 并把产出的 op 收集起来，
   预算耗尽或 batch 满就停。四类分支：Prefilling→`schedulePrefill`、
   Submitted/PrefetchDone→`schedulePrefillFirstChunk`、PrefillDone/Decoding→
   `scheduleDecode`、Retracted→`scheduleDecodeFromRetracted`。
3. **OOM 兜底**：radix 路径下若一轮什么都没排上（`ops.empty()`），说明显存耗尽，
   挑 token 最多的 decode 请求抢占（`newRetractOperation`，`forward.cpp:1224`）。
   flat 路径走 `resolveFlatStarvation`（`forward.cpp:1221`），需要连续两轮饥饿才放
   受害者，避免误触发。

### 调度门（gate）：以 schedulePrefillFirstChunk 为例（`operations/forward.cpp:291`）

每个门都是"能不能排这个请求"的可行性检查，**不通过就返回 `nullopt`、不改任何状态**：

1. 检查 req-pool 还有空槽（`req_pool_allocator_.AvailableSlots()`）。
2. **prefix 匹配**：`hybrid_prefix_cache_->Match()`（或裸 `kv_prefix_cache_.Match()`），
   算出 device / host 各命中多少页，从而得出这一轮真正要算的 token 数（`unscheduled`）
   和要从 host 换入的 token 数（`loadback_tokens`）。
3. 算需要的 device 页数，先 `EnsureCapacityByEvict<Device>()` 驱逐够空间
   （`forward.cpp:328`）——驱逐的是**没被 pin 的** prefix 缓存节点。
4. 若配了 mamba / paged-cache，还要过 `EnsureMambaCapacityByEvict` 和
   `hybrid_prefix_cache_->AdmitChunk`（`forward.cpp:379`，对 paged-cache 组做容量准入，
   用 `simulated_free` 在一轮内模拟多个请求的累计占用）。
5. flat 路径：`matchFlatPrefixAtAdmission` + `flatAdmit`（`forward.cpp:410`），走
   `KvCacheCoordinator::Admit` 做多组的驱逐 + 分配规划。

`scheduleDecode`（`forward.cpp:542`）/ `scheduleDecodeFromRetracted`（`:622`）/
`scheduleRetract`（`:710`）是同构的门，各自处理"续一个 decode token"、"从抢占恢复"、
"把一个请求踢出去换入 host"的可行性与资源准备。

### ExecutionPlan（`scheduler/execution_plan.h:46`）与 operations

`ExecutionPlan` 是一个 `Operation`（variant）列表。两大类：

- **`FlatForwardOperation`**（`operations/forward.h:87`）——把本轮所有 prefill/decode
  op 合并成 **SoA（struct-of-arrays）** 布局：`request_ids` / `request_pool_indices` /
  `input_lengths` / `input_ids` / `occupied_pages` / 每组的 `flat_block_tables` 等，
  都是并列的数组，方便 Python 直接建张量。构造函数（`forward.h:130`）用
  `stable_partition` 把 prefill 排在 decode 前，再逐字段展开；`flat_block_tables_contig`
  是行主序连续 buffer，Python 侧零拷贝成 2D ndarray（`python_module.cpp:387`）。
- **`CacheOperation`**——`FlatWriteBackOperation`（D2H，含抢占标记 `is_retract`）、
  `FlatLoadBackOperation`（H2D）、`PrefetchOperation`、`BackUpOperation`（L3）。
  每个带 `op_id`，Python 跑完异步搬运后用对应的 `*DoneEvent` 回灌，调度器据此推进 FSM。

`ForwardOperationBase`（`operations/forward.h:35`）的字段注释解释了 radix 与 flat 两种
page-table 表达的区别：radix 用 `paged_cache_pages`（compact + base_offset），flat 用
`flat_block_tables`（绝对逻辑页、null 洞=0、不 compact）。

### 外部事件回灌（`outside_event_handler.cpp`）

`handleEvent` 各重载把 Python 送回的完成事件翻译成 FSM 事件：

- `forward::ExtendResult`（`:205`）——新 token 到货，`ExtendResultEvent` 追加并可能
  publish 满页；flat 路径递减 `pending_forward_results_`（在途 forward 计数，
  `flatPoolWedged` 判断据此）。
- `forward::Finish`（`:170`）——`FinishEvent`，进 Draining（要 writeback）或直接 Finished。
- `forward::Abort`（`:222`）——`AbortEvent`，丢弃 KV（不进 prefix 缓存），用于数值损坏的请求。
- `cache::WriteBackDone`（`:244`）/ `LoadBackDone`（`:281`）——搬运完成，推进
  WritingBack→Finished / Retracting→Retracted，或释放 flat 的 store/load ticket。
- `cache::PrefetchDone`（`:47`）——host prefetch 完成，把 host 页插进 prefix 缓存。
- `pd::*`（`:99-168`）——PD 分离下的 bootstrap / 成功 / 失败 / 远端 prefill 完成。

## 四、radix 资源层（`csrc/resource/`，默认路径）

### 分配器（`resource/allocator/`）——自底向上

- **`PageAllocator`**（`page_allocator.h:30`）——物理页的 LIFO free-list。
  `Allocate(n)` 从尾部弹 n 个页 id 返回 `OwnedPages`，不够就返回空
  （`page_allocator.cpp:41`）。**页 id 从 1 开始，0 是保留的 dummy 页**
  （`page_allocator.cpp:36`）。
- **`OwnedPages`**（`owned_pages.h:31`）——move-only 的 RAII 页句柄，析构时把页 id
  还回它的 `PageAllocator`（`owned_pages.cpp:30`）。`TakeFirst/TakeLast` 切子句柄、
  `Append` 合并、`Detach` O(1) 交出 id 但**不**释放（给 Python 用）。这是上层所有
  分配器复用的唯一 RAII 原语。
- **`LocalKVAllocator`**（`kv_allocator.h:32`）——每个请求的**非 prefix 本地尾页**分配器。
  `Acquire(num_tokens)` 先填尾页、不够再向 `PageAllocator` 要新页
  （`kv_allocator.cpp:36`）。`TakeFullPages()` 交出除尾页外的满页去喂 radix 树。
- **`ReqPoolAllocator` / `ReqPoolIndex`**（`req_pool_allocator.h:54` / `:30`）——分配
  Python "请求池"的行号，FIFO；**slot 0 保留以对齐 Python 的 1-based 索引**
  （`req_pool_allocator.cpp:57`）。`ReqPoolIndex` 是 move-only RAII 句柄，析构即归还槽位。
- **`PagedCacheGroupAllocator` / `PagedCacheGroupTable`**（`paged_cache_group.h:85` / `:127`）
  ——模型自定义的"分页 cache 组"（如额外的 side cache）。前者是组级页池
  （Python 直接可用，`python_module.cpp:213`），后者是每个 (请求, 组) 的页表，分
  `borrowed_page_ids_`（物理页归 TreeNode snapshot 所有）和 `owned_pages_`（自己 RAII 持有）
  两段。这两个类连同 `PagedCacheGroupConfig`（`:55`）/ `Family{History,State}`（`:43`）
  都直接暴露给 Python，是 paged-cache 特性的地基。
- **Mamba 系**（`mamba_chunk_allocator.h` / `local_mamba_allocator.h` /
  `mamba_host_allocator.h`）——SSM 状态槽位池。用**最小堆** free-list，`Allocate` 永远
  返回最小空闲下标，与 `Free` 顺序无关——这是 **TP rank 间确定性**的关键
  （释放回调在各 rank 上乱序触发）。

### radix 树与 prefix 缓存（`resource/radix_tree/`、`kv_prefix_cache/`、`hybrid_prefix_cache/`）

- **`TreeNode`**（`radix_tree/tree_node.h:53`）——按页对齐 token 串为 key 的 radix 节点，
  挂最多五个可拆卸附属物：device/host KV 资源、mamba device/host 槽、paged-cache snapshot。
  带单调 `seq_id_`（TP 确定性 tiebreaker）和 LRU 时间戳。
- **`RadixTree`**（`radix_tree/radix_tree.h:50`）——prefix 匹配引擎。
  `WalkDownUtilMismatch`（`radix_tree.cpp:134`）逐页从 root 往下走，遇部分匹配就
  `splitChild` 裂节点，独立跟踪 device 和 host 两级的最深命中；`PruneEmptyByNode`
  从叶往根删空节点，删前触发 `node_destroy_callback_` 让附属物先解引用。
- **`KVPrefixCache`**（`kv_prefix_cache/kv_prefix_cache.h:44`）——厂商中立的 **KV-only**
  两级（device/host）prefix 缓存。`Match`（`kv_prefix_cache.cpp:163`）做匹配，
  `Insert<RType>`（`:199`）把满页登记进树并挂上分配器页，`EnsureCapacityByEvict<RType>`
  （`:343`）按 LRU 驱逐未 pin 的叶子。纯 Transformer 模型走这条。
- **`HybridPrefixCache`**（`hybrid_prefix_cache/hybrid_prefix_cache.h:49`）——**组合**
  （不是替换）一个 `KVPrefixCache&`，在**同一批 radix 节点**上叠加两个正交的附属系统：
  ① Mamba/SSM 状态缓存（独立的 `MambaEvictionManager` + 自己的 LRU）；② 多组
  paged-cache（快照 pin 到 TreeNode 上）。hybrid SSM/Mamba 或用 paged-cache 的模型走这条。
  调度器构造时把两级驱逐回调 + 节点销毁回调都接到它上面（`scheduler.cpp:124-133`），
  保证节点被删前所有附属物先干净解引用。
- **`page_hasher.h`**——内容寻址的页 hash。`HashPage`（`:89`）用 **prefix-framed
  SHA256 链**：`[prior_len][prior_hash][token_count][tokens][extra…]`，让
  `(prior, tokens, extra)` 三元组无歧义；`ComputePagedHashes`（`:120`）把第 N 页 hash
  当第 N+1 页的 prior，形成 Merkle 式链。这些 hex 串填进 `TreeNode::page_hashes_`。

## 五、flat 资源层（`csrc/cache/`，`TOKENSPEED_FLAT_KVCACHE=ON`）

面向 flat-state / LCM two-level 布局（详见 `kvcache-management.md`）。整棵 `csrc/cache/`
只在 flat build 编译。

- **`KvCacheCoordinator`**（`cache/kv_cache_coordinator.h:48`）——多个 cache 组共享一个
  device `BlockPool`（+ 可选 host pool）的门面。不持有 per-request 状态，发全局单调的
  `access_epoch`（LRU 时钟）。`ProbePrefix`（`:93`）只读地算跨组最长公共 prefix；
  `Admit`（`:98`，实现在 `kv_cache_admission.cpp:288`）消费 probe、规划并提交驱逐 +
  分配，返回 `AdmissionResult`（device/host prefix token 数、load pairs、new_page_ids）
  或 `nullopt`。`MakeCoordinator`（`:157`）是工厂，按每个 `KvCacheSpec` 的 kind 建对应
  Manager。
- **attention Manager 家族**——每组一个，插到 coordinator 上：
  - `KvCacheManager`（`kv_cache_manager.h:44`）抽象基类，管 per-组 token 策略 + cache-entry
    索引；`ResolveKernelPageId` 把 LCM (block,slot) 折算成 kernel 用的物理页 id。
  - `FullAttnManager`（`full_attn_manager.h:35`）——full attention，闭合 prefix，左→右
    走到首个 miss。
  - `SwaManager`（`swa_manager.h:35`）——sliding window，非闭合，右→左找"能恢复的边界"，
    滑出窗口的块打成 null 洞保持 slot 对齐。
  - `MambaStateManager`（`mamba_state_manager.h:32`）——`SwaManager` 特化（窗口=2，
    只留一份 live 状态页）。
- **`BlockPool`**（`cache/block_pool.h:38`）——纯物理 LCM-block 分配器，不管 key/LRU/
  所有权。`AcquireBlocks`（`:62`）K=1 时弹 free-parent 队首，K>1 时用 `planLocations`
  优先填最满的已绑定 parent。
- **`AdmissionPlanner`**（`kv_cache_admission.cpp:40`）——"零驱逐优先"的准入规划：先数
  空闲槽 + 空 parent，装不下就按 `EvictionTier`（`:99`：kUncached < kProbationaryBoundary
  < kEstablishedBoundary < kClosedPrefix）+ LRU epoch 建受害者最大堆，弹到够用为止，
  再尽量恢复非必需的。**保护当前 prefix 命中的块不被驱逐**。
- **`CacheBlockRef`**（`cache_block_ref.h`）——侵入式引用计数的池槽句柄，`use_count()`/
  `unique()` 是各处判断"能否驱逐"的依据。
- **`forward_cache_ops.cpp`**——config↔coordinator 的桥：`MakeSpecsFromConfig`（`:50`）把
  `PagedCacheGroupConfig` 映射成 `KvCacheSpec`，`BuildFlatBlockTables`（`:105`）产出每组的
  kernel 页 id 行，`AlignFlatPrefillChunk`（`:30`）把 prefill 分块对齐到页/promotion 边界。

## 六、构建与测试

- **构建**：`pyproject.toml` 用 scikit-build-core，`CMakeLists.txt` 用 nanobind
  （`:99`）+ tokenspeed-spdlog（`:19-29`）+ OpenSSL（SHA256，`:36`）。默认打
  `TOKENSPEED_SCHEDULER_BUILD_PYTHON=ON`、tests OFF、flat-kvcache OFF
  （`pyproject.toml:25`、`CMakeLists.txt:10-12`）。C++ 核心先编成静态库
  `tokenspeed_scheduler_core`（`CMakeLists.txt:38`），再链进扩展。
- **C++ 测试**（`tests/cpp/`，41 个 gtest，`CMakeLists.txt:126` 起）——覆盖生命周期
  （`test_basic_lifecycle`）、抢占（`test_retract*`）、chunked prefill、batch 调度、
  LRU 驱逐、prefetch、write/load back、flat kvcache（`test_flat_kvcache_*`）、mamba
  （`test_mamba_*`）等。用 `TOKENSPEED_SCHEDULER_BUILD_TESTS=ON` + FetchContent 拉
  googletest。
- **Python 测试**（`python/tests/`，10 个）——`test_cache_binding` / `test_fsm_and_scheduling`
  / `test_paged_cache_group*` / `test_pd_binding` / `test_flat_kvcache` / `test_req_pool_indices`
  等，验证绑定层和端到端调度语义。

## 七、几个关键设计约束

- **TP 确定性**：多处（候选排序按 `Id()`、mamba 分配用最小堆、LRU 用 `seq_id` tiebreak）
  都是为了让 TP 各 rank 做出**逐位一致**的调度决策——否则 NCCL 集合通信会因 op 不一致
  而死锁。这是分布式推理的硬约束。
- **调度器不碰显存**：它只算逻辑页 id 和 op，真正的 GPU KV buffer、H2D/D2H 拷贝、清零
  都在 Python runtime（见 `kvcache-management.md` 的 MemoryExecutor 章节）。页 id 0
  在两条路径下都是 null 占位符。
- **radix / flat 编译期二选一**：不是运行时开关。同一份源码两种 build，Python 靠
  `FLAT_KVCACHE` 常量适配。新特性（LCM two-level、flat PD）都在 flat 路径上演进，
  radix 是当前默认与回退。
- **依赖边界**：本子项目是 `tokenspeed` runtime 依赖的独立 wheel（`tokenspeed-scheduler`），
  与 vendor-neutral 原则一致——它不含任何厂商 kernel，只做纯 CPU 的调度决策。
