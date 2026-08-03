# tokenspeed-scheduler 目录详解
##### commit-id: 638b9de0e698446b5e50a2a9508778b2f59473f6

本文详细介绍 `tokenspeed-scheduler/` 这个子项目：它是 TokenSpeed 的**核心调度器**，
一个用 C++20 写、通过 nanobind 暴露给 Python 的独立扩展模块。Python 侧的
`event_loop.py` 每个迭代都调用它来决定"这一步 forward 跑哪些请求、每个请求占哪些
KV 页、要不要做 prefix 复用 / 换入换出 / 抢占"。它是 `inference-flow.md`
（一次请求端到端流程）里 "scheduler subprocess: RequestHandler + C++ Scheduler"
那一环的展开，也是 `kvcache-management.md` 里"分配决策（C++ scheduler）"层的完整实现。

> **重大变更（PR #864 "Remove Radix cache path and consolidate runtime cache
> handling"）**：旧的 **radix / flat 编译期双后端已删除**，radix 路径连同整个
> `csrc/resource/radix_tree/`、`kv_prefix_cache/`、`hybrid_prefix_cache/`
> 目录（约 11k 行）被移除。现在**只有 flat（paged-cache）一条路径**，`csrc/cache/`
> 是唯一资源层。CI 用 `tests/check_no_radix_cache_sources.py` 守护这些符号/目录不再
> 出现。若你手上的旧文档还在讲 `TOKENSPEED_FLAT_KVCACHE` 开关或 `RadixTree` /
> `KVPrefixCache`，那些都已不存在。

> 约定：全文用大量 `path:line` 锚点，路径相对 `tokenspeed-scheduler/`。行号会随无关
> 改动漂移，更新本文时请逐个用 `grep -n` / `sed -n` 核对。

## 总览

调度器只做**决策**，不持有 GPU 显存。它维护每个请求的状态机（FSM），在每个迭代里
挑一批请求、算好它们要占的逻辑页，产出一份 `ExecutionPlan` 交给 Python runtime 去真正
跑 forward、搬 KV。

```text
Python event_loop.py（每个迭代）
  │
  │ submit_requests(specs)        ← 新请求进入（Submitted / Bootstrapping 状态）
  ▼
┌─ C++ Scheduler ────────────────────────────────────────────────────┐
│  requests_: id → Request（每个请求一个 FSM）                         │
│  block_pool_ / host_pool_（物理 LCM block 池）                       │
│  coordinator_（多 cache group 门面）+ tier_transfers_（D2H/H2D）     │
│                                                                      │
│  next_execution_plan():                                            │
│    1. 回收 Finished，给待写回的块生成 WriteBack op                   │
│    2. buildForwardOperations(candidates):                          │
│         按优先级排序（同优先级按 Id() → TP 各 rank 一致）            │
│         逐个请求过"调度门"(schedulePrefillFirstChunk/Decode…)       │
│         每个门：prefix 匹配 + 容量准入(Admit/驱逐) → 通过则 Apply    │
│         事件推进 FSM 并分配 block，产出 Prefill/Decode Operation     │
│    3. 打包成 ExecutionPlan（forward ops + cache ops + pages_to_zero）│
│  ▲                                                                   │
│  │ advance(ExecutionEvent)      ← forward/PD/cache 完成事件回灌      │
│  │   ExtendResult / Finish / Abort / WriteBackDone / LoadBackDone…  │
└──┼──────────────────────────────────────────────────────────────────┘
   │ next_execution_plan() 返回
   ▼
Python runtime：清零新页 → 提交 cache op（H2D/D2H）→ 跑 forward → 采样
                → 把结果 advance() 回灌调度器
```

资源层只有一套（flat / paged-cache）：物理页由 `BlockPool` 持有，
`KvCacheCoordinator` 在其上对每个 cache group 各挂一个 `KvCacheManager`
（full-attention / sliding-window / mamba-state），prefix 匹配是每个 manager 在自己
的 cache-entry 索引上做 `Probe`，不再有独立的 radix 树或 prefix-cache 类。面向
flat-state / LCM two-level 布局（见 `lcm.md` 与 `kvcache-management.md`）。

## 目录结构

```text
tokenspeed-scheduler/
├── CMakeLists.txt          构建（scikit-build-core + nanobind + spdlog + OpenSSL）
├── pyproject.toml          包名 tokenspeed-scheduler（v0.1.5），wheel 打 python/tokenspeed_scheduler
├── bindings/
│   └── python_module.cpp   nanobind 绑定：唯一的 Python↔C++ 边界
├── csrc/
│   ├── core/               token_container：请求 token 序列 + 分页视图
│   ├── fsm/                请求状态机：states / forward_{states,events} / pd_{states,events}
│   ├── resource/allocator/ req_pool_allocator（resource/ 下唯一幸存者）
│   ├── cache/              唯一资源层（flat / paged-cache）
│   │   ├── core/           block_pool / block_table / cache_block_ref / cache_config / cache_types
│   │   ├── coordinator/    kv_cache_coordinator + kv_cache_admission（AdmissionPlanner）
│   │   ├── manager/        kv_cache_manager 基类 + full_attn / swa / mamba_state / cache_group
│   │   └── tier/           transfer / transfer_manager（D2H/H2D 异步搬运）
│   ├── scheduler/          调度主逻辑：Scheduler / Request / ExecutionPlan / operations/ / outside_events/
│   └── utils.h
├── python/
│   ├── tokenspeed_scheduler/__init__.py   Python 包（re-export 扩展符号）
│   └── tests/              7 个 pytest（cache 绑定 / FSM / paged-cache / PD…）
└── tests/
    ├── cpp/                16 个 gtest（生命周期 / 驱逐 / 抢占 / coordinator / swa…）
    └── check_no_radix_cache_sources.py   CI 守护：radix 目录/符号不得复活
```

## 一、Python API 边界（`bindings/python_module.cpp`）

整个扩展只有一个 nanobind module `tokenspeed_scheduler_ext`（`python_module.cpp:72`），
`python/tokenspeed_scheduler/__init__.py` 只是把符号 re-export 出来（不再导出
`FLAT_KVCACHE` 常量）。核心暴露：

- **`Scheduler`**（`python_module.cpp:304`）——唯一有状态的对象。方法映射：
  - `submit_requests(specs)` → `SubmitRequests`（`:306`）
  - `next_execution_plan()` → `NextExecutionPlan`（`:309`）
  - `advance(event)` → `Advance`（`:310`）
  - `drain_kv_events()`（`:311`）——排空 KV-cache 的 BlockStored/BlockRemoved 事件
  - 一堆只读监控：`waiting_size` / `decoding_size` / `prefilling_size` /
    `available_kv_pages` / `active_kv_pages` / `paged_cache_group_*_pages`（`:319`-`330`）
- **`SchedulerConfig`**（`python_module.cpp:87`，结构体在 `scheduler/types.h:31`）——
  一次性传入的配置：`block_size`、`max_scheduled_tokens`、`max_batch_size`、
  `role`（P/D/Fused，enum 在 `:89`）、`host_allocator`/`device_allocator{total_pages}`、
  `paged_cache_groups`（`types.h:42`）、各种开关（`disable_prefix_cache` `:65` /
  `enable_mixed_prefill_decode` `:60` / `enable_pd_cache` / `disable_l2_cache` /
  `enable_l3_storage` / `prefix_replay_tokens` `:69`）。`HasHostCache()` `:44` /
  `StreamsDeviceCacheToHost()` `:48` 是派生判断。
- **`PagedCacheGroupConfig`**（`python_module.cpp:107`）+ 三个 enum：`PagedCacheRetention`
  （`:94`）、`PagedCacheGroupFamily`（`:98`）、`PagedCacheTransferPolicy`（`:102`）——
  描述每个 cache group 的 retention / family（history/state）/ PD 传输策略。
- **`RequestSpec`**（`python_module.cpp:164`，结构体 `scheduler/request_spec.h:30`）——
  提交一个请求需要的最小信息：`request_id` + `tokens` + 可选的 L3 storage 命中信息。
- **输入事件**（Python → 调度器，通过 `advance`）：
  - `ForwardEvent.{ExtendResult, Finish, Abort, UpdateReserveNumTokens}`（`:170`-`184`）
  - `Cache.{WriteBackDoneEvent, LoadBackDoneEvent}`（`:195` / `:199`）
  - `PD.{BootstrappedEvent, FailedEvent, SucceededEvent, RemotePrefillDoneEvent}`（`:203`-`215`）
- **输出**（调度器 → Python）：`ExecutionPlan`（`:298`），暴露 `forward`（一批
  `Forward.Batch`）、`cache`（`Cache.LoadBackOp` `:266` / `Cache.WriteBackOp` `:272`）、
  `pages_to_zero`。`Forward.Batch`（`:231`）暴露 SoA 字段与零拷贝
  `block_tables_arrays()`（`:247`）。

`ExecutionEvent`（`:220`）是事件容器，Python 用 `add_event`（`:223`）往里塞多个事件，
一次 `advance` 全部消费。

## 二、请求状态机（`csrc/fsm/`）

每个请求是一个 `Request`（`scheduler/request.h:41`），内部持有一个
`fsm::State`——**7 个状态**的 `std::variant`（`fsm/states.h:31`，旧的 13 状态精简而来，
cache-FSM 子状态已随 radix 删除折进 coordinator/tier-manager）：

```text
Bootstrapping ─(PD only, Bootstrapped)→ Submitted
Submitted ─prefill首块→ Prefilling ─(还有块)→ Prefilling
                             └─(最后块)→ PrefillDone ─decode→ Decoding ─decode→ Decoding
任意 forward 态 ─Finish→ Finished（要写回时先在 coordinator 侧排 WriteBack op）
Decoding ─抢占(轻)→ Retracted ─(恢复)→ Decoding      （RetractionEvent：释放 device，留 L2）
{PrefillDone,Decoding} ─抢占(重)→ Submitted           （RetractEvent：全释放 + 重排队）
任意态 ─Abort→ Finished
```

状态定义（`fsm/forward_states.h`，PD 的 `Bootstrapping` 在 `fsm/pd_states.h:29`）：
`Submitted`（`:65`）、`ForwardState` 基类（`:77`，持 `req_pool_index` / `block_tables`
/ `CacheProgress`）、`Prefilling`（`:126`，含 `PrefillSource` `:38` kLocal/kRemote）、
`PrefillDone`（`:158`）、`Decoding`（`:186`）、`Retracted`（`:207`）、`Finished`（`:215`）。
`CacheProgress`（`:40`）记 `page_hashes` / `access_epoch` / `promotion_boundary_tokens`。

设计要点：

- **事件 = 转移函数**。base CRTP `InvalidTransitionHandler<Derived>`（`fsm/base_event.h:32`）
  兜底非法转移抛异常。`Request::Apply`（`request.h:47`）用 `std::visit` 把当前状态喂给
  事件的 `operator()`，返回新状态；`Request::Is<State>()`（`:63`）查状态。
- **forward 事件**（`fsm/forward_events.h` 声明、`forward_events.cpp` 实现转移）：
  `SchedulePrefillFirstChunkEvent`（`:40`，Submitted/Retracted→Prefilling/PrefillDone）、
  `SchedulePrefillEvent`（`:79`，续块）、`ScheduleDecodeEvent`（`:96`）、`FinishEvent`
  （`:113`，→Finished，调 `FreeRequest`）、`AbortEvent`（`:130`，任意态→Finished）、
  `RetractionEvent`（`:152`，Decoding→Retracted，释放 device 保留 L2/store）、
  `RetractEvent`（`:164`，{PrefillDone,Decoding}→Submitted 全释放 + 重排队）、
  `UpdateReserveNumTokensEvent`（`:179`）、`ExtendResultEvent`（`:194`，追加 forward 出的
  新 token）。
- **PD 事件**（`fsm/pd_events.h` / `pd_events.cpp`）：`BootstrappedEvent`（`:33`，
  Bootstrapping→Submitted）、`SucceededEvent`（`:39`，Decoding→Finished）、
  `RemotePrefillDoneEvent`（`:45`，Prefilling→PrefillDone，并把 prefill 采出的 bootstrap
  token 追加进 token 容器）。

`StateName()`（`request.h:121`）给出可读状态名，调试和错误信息都用它。

## 三、调度主循环（`csrc/scheduler/`）

### Scheduler 类（`scheduler.h:50`，实现分散在 `scheduler.cpp` + `operations/`）

构造函数（`scheduler.cpp:68`）按 config 建好所有资源：`req_pool_allocator_`（`:160`）、
`block_pool_`（`:163`，`total-1` 个 device parent）、`host_pool_`（`:164`）、
`coordinator_ = MakeCoordinator(MakeSpecsFromConfig(config_), …)`（`:165`）、
`tier_transfers_`（`:166`）。还算 `max_single_request_tokens_`（二分，`calculateMax
SingleRequestTokens` 在 `:191`）。关键嵌套结构：`AdmissionMatch`（`:88`）、
`PlanBuildContext`（`:99`，携带 `admission_failed` / `waits_for_store_ack` /
`capacity_blocker`）。

三个入口方法：

1. **`SubmitRequests`**（`scheduler.cpp:325`）——把每个 `RequestSpec` 包成 `Request`
   （Fused 角色为 Submitted，PD decode 角色为 Bootstrapping，见 `request.cpp:29`）塞进
   `requests_`。
2. **`NextExecutionPlan`**（`scheduler.cpp:415`）——一个迭代的核心：先 erase Finished
   （`:416`），收集 `{Submitted,Prefilling,PrefillDone,Decoding,Retracted}` 候选
   （`:426`-`431`），调 `buildForwardOperations`（`:436`）得到本轮 forward ops +
   loadback ops，包进 `ExecutionPlan` 并追加 WriteBack/LoadBack `CacheOperation`
   （`:439`-`449`）。
3. **`Advance`**（`scheduler.cpp:453`）——双层 `std::visit` 分发 `ExecutionEvent`
   里的每个事件到对应的 `handleEvent` 重载（实现全在 `outside_event_handler.cpp`）。

### 核心：buildForwardOperations（`operations/forward.cpp:511`，旧名 `newForwardOperation`）

这是"这一步跑哪些请求"的决策心脏（声明 `scheduler.h:108`）：

1. **恢复队列/屏障维护**（`:514`-`526`）：处理 retraction 恢复的顺序。
2. **优先级排序**：优先级 lambda（`:527`-`554`）——recovery-front=0、barrier=1、
   retracted-recovery=2、Prefilling=3、Submitted=4、Decoding/PrefillDone=5（开
   `enable_mixed_prefill_decode` 时 decode 提到 3）。`std::ranges::sort`（`:555`）后
   **同优先级按 `Id()` 排序**——保证 TP 各 rank 调度完全一致的子集（不一致会让 NCCL
   死锁）。
3. **token 预算循环**（主候选循环 `:601`-`685`）：逐个候选过对应的"调度门"，通过就
   `Apply` 事件推进 FSM 并把产出的 op 收集起来（`push_operation` 在 `:576` 做预算/
   batch 记账），预算耗尽或 batch 满就停。
4. **抢占兜底**（`:687`-`689`）：一轮排不上（容量耗尽）就 `retractForCapacity`
   （`:450`）挑受害者抢占，`beginRetraction` 在 `:433`。

### 调度门（gate）与准入（`operations/forward.cpp`）

每个门都是"能不能排这个请求"的可行性检查，**不通过就不改任何状态**：

- `schedulePrefillFirstChunk`（`:251`，声明 scheduler.h:110）/ `schedulePrefill`
  （`:327`）/ `scheduleDecode`（`:370`）——分别处理"首块 prefill"、"续块"、"续一个
  decode token"。通过后各自的 `applyEventAndBuildOperation` 重载（`:406` / `:418` /
  `:422`）Apply 事件并产出 `PrefillOperation` / `DecodeOperation`。
- **prefix 匹配 + 准入**：`matchPrefixAtAdmission`（`:167`）在 coordinator 上算跨组
  最长公共 prefix；`admit`（两个重载 `:210` / `:236`）消费 probe，调
  `KvCacheCoordinator::Admit` 做多组的驱逐 + 分配规划；`admitWithKvEventTracking`
  （`:242`）在准入时附带 KV 事件记账。

### ExecutionPlan（`scheduler/execution_plan.h:33`）与 operations

`ExecutionPlan` 持有一个 `Operation`（variant）列表，`Operation = variant<CacheOperation,
ForwardBatch>`（`operations/inc.h:30`），外加 `pages_to_zero` map（`execution_plan.h:46`，
新分配、forward 前要清零的 per-group 页）。`With()` 在 `:36`，`Operations()` 在 `:41`。

- **`ForwardBatch`**（`operations/forward.h:60`）——把本轮所有 prefill/decode op 合并成
  **SoA（struct-of-arrays）** 布局，Python 侧直接建张量，并通过 `block_tables_arrays()`
  零拷贝拿每组的 2-D int32 page-table 视图。`ForwardOperation = variant<PrefillOperation,
  DecodeOperation>`（`:58`），`PrefillOperation` 在 `:47`、`DecodeOperation` 在 `:54`、
  `ForwardOperationBase` 在 `:36`。
- **`CacheOperation`**（`cache/tier/transfer.h:122`）——`LoadBackBatch`（H2D，`:94`）/
  `WriteBackBatch`（D2H，`:61`）。每个带 op 标识，Python 跑完异步搬运后用
  `Cache.*DoneEvent` 回灌，调度器据此推进。

### 外部事件回灌（`outside_event_handler.cpp`）

`handleEvent` 各重载（9 个）把 Python 送回的完成事件翻译成 FSM 事件：

- `pd::Bootstrapped`（`:37`）/ `pd::Failed`（`:44`）/ `pd::Succeeded`（`:54`）/
  `pd::RemotePrefillDone`（`:67`）——PD 分离下的 bootstrap / 失败 / 成功 / 远端 prefill
  完成（详见 `pd-disaggregation.md`）。
- `forward::Finish`（`:87`，内部 `publishCompletedPages` 在 `:100`，把满页登记进
  cache）/ `forward::UpdateReserveNumTokens`（`:122`）/ `forward::ExtendResult`（`:128`）
  / `forward::Abort`（`:138`，丢弃 KV 不登记，用于数值损坏的请求）。
- `cache::WriteBackDone`（`:146`）/ `cache::LoadBackDone`（`:150`）——搬运完成，推进
  tier transfer 的 store/load ticket。

## 四、资源层（`csrc/cache/`，唯一路径）

架构：**一个 device `BlockPool` + 可选 host `BlockPool`** 物理持有 LCM block；
**`KvCacheCoordinator`** 在其上对 N 个 **`CacheGroup`** 做门面，每组包一个
**`KvCacheManager`** 子类；**`AdmissionPlanner`** 做驱逐规划；**`TierTransferManager`**
做异步 D2H/H2D。没有 radix 树、没有独立 prefix-cache 类——prefix 匹配是每个 manager 在
自己的 cache-entry 索引上做 `Probe`。

### cache/core/

- **`BlockPool`**（`cache/core/block_pool.h:38`）——纯物理 LCM-block 放置，不管 key/LRU/
  所有权。`AcquireBlock`/`AcquireBlocks`（`:54`/`:62`，K>1 时用 `planLocations` `:191`
  优先填最满的已绑定 parent）、`Release`（`:125`）、`BoundGroup`（`:94`）、
  `NumEmptyLcmBlocks`（`:52`）。内部 `LcmBlock` 在 `:145`。
- **`BlockTable`**（`cache/core/block_table.h:34`）——每请求逻辑→物理页映射，`FromBlocks`
  （`:36`）、`Blocks`（`:44`）、`EvictToNull`（`:48`）。
- **`CacheBlockRef` / `CacheBlock` / `CacheBlockControl`**（`cache/core/cache_block_ref.h`，
  `:96`/`:49`/`:68` + `.cpp`）——侵入式引用计数的池槽句柄，`use_count()`/`unique()` 是各处
  判断"能否驱逐"的依据；`CacheBlockLocation{lcm_block_id, slot_index}` 在 `:34`。
- **`cache_types.h`**：`enum AttnKind {kFull,kSlidingWindow,kMambaState}`（`:37`）、
  `CacheKey`（`:48`，`{namespace_id,group_id,content_hash,cache_block_offset}`）、
  `KvCacheSpec`（`:70`，`{kind,sliding_window,cache_blocks_per_lcm_block,cache_block_tokens}`）、
  `GroupDemand`（`:89`，per-group 准入输入）、`PrefixMatch`（`:103`）。
- **`cache_config.{h,cpp}`**：`PagedCacheGroupConfig`（`:37`，`CacheBlockTokens()` `:54`，
  `Validate()` 在 `.cpp:27`）、`PagedCacheGroupFamily{History,State}`（`:29`）、
  `PagedCacheTransferPolicy{Unspecified,FullSuffix,LatestSnapshot}`（`:31`）。

### cache/coordinator/

- **`KvCacheCoordinator`**（`cache/coordinator/kv_cache_coordinator.h:51`）——多组 cache
  的门面，持 `vector<CacheGroup> groups_`、`BlockPool& pool_`、`BlockPool* host_pool_`、
  单调 `next_access_epoch_`（LRU 时钟）、`pending_stores_`。`enum CacheTier{kDevice,kHost}`
  （`:40`）、`PrefixProbe`（`:70`）、`AdmissionResult`（`:83`）。关键方法：`ProbePrefix`
  （`:281`，只读算跨组最长公共 prefix）、`Admit`（实现在 `kv_cache_admission.cpp:316`）、
  `CanAdmitAfterReleasing`（admission.cpp:301）、`Free`（`:626`）、`ClearDeviceCache`/
  `ClearCache`（`:79`/`:99`）、`CacheFullBlocks`（`:407`）、`AcquireHostBlock`/
  `CacheHostBlock`（`:489`/`:671`）。工厂 **`MakeCoordinator`**（`kv_cache_coordinator.h:202`
  / `.cpp:677`）按每个 `KvCacheSpec` 的 kind 建对应 Manager。
- **`kv_cache_admission.cpp`**——eviction/capacity 规划。`AdmissionPlanner`（`:40`，
  `EvictionTier{kUncached,kProbationaryBoundary,kEstablishedBoundary,kClosedPrefix}` 在
  `:103`，`Plan()` 在 `:56`）"零驱逐优先"：先数空闲槽 + 空 parent，装不下再按 tier +
  LRU epoch 建受害者最大堆弹到够用，**保护当前 prefix 命中的块不被驱逐**。

### cache/manager/

- **`KvCacheManager`**（抽象基类，`cache/manager/kv_cache_manager.h:44`）——管 per-组
  token 策略 + cache-entry 索引（`CacheEntries` 在 `:463`，`by_key`/`by_location` 映射进
  `std::list`）。虚函数 `MatchIsPrefixClosed`（`:90`）/ `Probe`（`:92`）/ `ReclaimExpired`
  （`:403`）；具体 `Acquire`（`:143`/`:219`）、`RegisterCachedBlock`（`:243`）、
  `EvictCachedBlock`（`:368`）、`ResolveKernelPageId`（`:71`，把 LCM (block,slot) 折算成
  kernel 物理页 id）。
- **`FullAttnManager`**（`full_attn_manager.h:35`）——full attention，闭合 prefix，左→右
  走到首个 miss。
- **`SwaManager`**（`swa_manager.h:35`）——sliding window，非闭合，右→左找可恢复边界，
  滑出窗口的块打成 null 洞保持 slot 对齐。
- **`MambaStateManager`**（`mamba_state_manager.h:32`）——`SwaManager` 特化（窗口=2，
  只留一份 live 状态页）。
- **`CacheGroup`**（`cache_group.h:32`）——spec + 拥有的 manager。

### cache/tier/（替代旧 forward_cache_ops）

- **`transfer.h`**：wire 类型 `CacheTransfer`（`:35`）、`WriteBackOperation`（`:56`）+
  `WriteBackBatch`（`:61`）、`LoadBackOperation`（`:89`）+ `LoadBackBatch`（`:94`）、
  `using CacheOperation = variant<LoadBackBatch, WriteBackBatch>`（`:122`）。
- **`TierTransferManager`**（`cache/tier/transfer_manager.h:38` + `.cpp`）——
  `StartPendingStores`（`.cpp:40`）、`StartPrefixLoad`（`.cpp:83`）、`CompleteWriteBack`/
  `CompleteLoadBack`（`.cpp:100`/`117`）、in-flight 谓词（`:48`-`50`）。

### resource/allocator/（resource/ 下唯一幸存者）

- **`ReqPoolAllocator` / `ReqPoolIndex`**（`resource/allocator/req_pool_allocator.h:57` /
  `:30`）——分配 Python "请求池"行号的 RAII 槽位分配器；**slot 0 保留以对齐 Python 的
  1-based 索引**（`.cpp:57`），`Allocate` 在 `.cpp:64`。

### 其它 scheduler/ 组件

- **`core/token_container.{h,cpp}`**——`TokenContainer`（`h:31`）；`RebasePrefill()`
  （`h:48`，retraction 时把已生成 token 并回 prefill 窗口）。
- **`scheduler/page_hasher.h`**——内容寻址的页 hash：`HashPage`（`:89`）用 prefix-framed
  SHA256 链，`ComputePagedHashes`（`:120`）把第 N 页 hash 当第 N+1 页的 prior，形成
  Merkle 式链。
- **`scheduler/kv_cache_events.{h,cpp}`**——`KvCacheEvent = variant<KvBlockStoredEvent,
  KvBlockRemovedEvent>`（`h:47`），`drain_kv_events()` 排空给 Python 上报。
- **`scheduler/operations/cache.{h,cpp}`**——free 函数：`MakeSpecsFromConfig`（`cpp:49`，
  `PagedCacheGroupConfig`→`KvCacheSpec`）、`AlignPrefillChunk`（`cpp:29`）、`FreeRequest`
  （`cpp:100`）、`BuildBlockTables`（`cpp:107`）。
- **`scheduler/outside_events/`**——Python-facing 事件结构：`forward.h`
  （`ExtendResult`/`Finish`/`UpdateReserveNumTokens`/`Abort` + `ForwardEvent` variant）、
  `pd.h`（`BootstrappedEvent`/`FailedEvent`/`SucceededEvent`/`RemotePrefillDoneEvent` +
  `PDEvent`）、`cache.h`（`WriteBackDone`/`LoadBackDone` + `CacheEvent`）。
  `Event = variant<CacheEvent, ForwardEvent, PDEvent>`（`outside_events/inc.h:33`）。

## 五、构建与测试

- **构建**：`pyproject.toml`（v0.1.5）用 scikit-build-core，`CMakeLists.txt` 用 nanobind
  （`:34`）+ tokenspeed-spdlog（`:18`-`28`）+ OpenSSL（SHA256，`:35`）。选项只剩
  `TOKENSPEED_SCHEDULER_BUILD_PYTHON`（`:10`）/ `TOKENSPEED_SCHEDULER_BUILD_TESTS`
  （`:11`）——**没有 flat-kvcache 开关**。C++ 核心先编成静态库
  `tokenspeed_scheduler_core`（`:37`），再链进扩展。
- **C++ 测试**（`tests/cpp/`，16 个 gtest，`CMakeLists.txt:106` 起）——覆盖生命周期、
  抢占（`test_retract*` 系并入 lifecycle/scenarios）、coordinator（`test_kv_cache_coordinator`）、
  swa（`test_swa_manager`）、outside event（`test_outside_event_handler`）、scheduler plan
  （`test_scheduler_plan`）等。用 `TOKENSPEED_SCHEDULER_BUILD_TESTS=ON` + FetchContent 拉
  googletest。
- **Python 测试**（`python/tests/`，7 个）——验证绑定层和端到端调度语义。
- **CI 守护**：`tests/check_no_radix_cache_sources.py` 若发现 `radix_tree` /
  `kv_prefix_cache` / `hybrid_prefix_cache` 目录或 `RadixTree` / `KVPrefixCache` /
  `HybridPrefixCache` 符号复活就 fail。

## 六、几个关键设计约束

- **TP 确定性**：多处（候选排序按 `Id()`、LRU 用 `access_epoch`/seq tiebreak）都是为了让
  TP 各 rank 做出**逐位一致**的调度决策——否则 NCCL 集合通信会因 op 不一致而死锁。这是
  分布式推理的硬约束。
- **调度器不碰显存**：它只算逻辑 block/page id 和 op，真正的 GPU KV buffer、H2D/D2H
  拷贝、清零都在 Python runtime（见 `kvcache-management.md` 的 MemoryExecutor 章节）。
  block/page id 0 是 null 占位符（parent 0 是永不调度的 null LCM block）。
- **单一 flat / paged-cache 路径**：radix 双后端已删除（PR #864）。所有 KV 都走
  `BlockPool` + `KvCacheCoordinator`，几何由 LCM two-level 布局给出（见 `lcm.md`）。
- **依赖边界**：本子项目是 `tokenspeed` runtime 依赖的独立 wheel（`tokenspeed-scheduler`），
  与 vendor-neutral 原则一致——它不含任何厂商 kernel，只做纯 CPU 的调度决策。
