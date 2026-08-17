# KV Cache 管理机制

本文描述当前生产路径。旧的 Python `KVAllocator`、radix `PrefixCache`、
`BaseTokenToKVPool`、`LcmCachePool` 与 `paged_cache_spec.py` 已不存在；allocation、
prefix reuse 与 block table 生命周期统一在 C++ `tokenspeed-scheduler` 的 cache 子系统。

## 总览

```text
CacheRecipe
  └─ CacheMemoryPlan + CacheGroupSpec[]
       └─ CacheArena (GPU bytes + typed field views + runtime contract)
            ├─ target/draft CachePool views
            └─ Python bridge: pool_to_cache_groups()
                    │
                    ▼
C++ SchedulerConfig.cache_groups
  └─ CacheCoordinator
       ├─ PrefixMatcher / PrefixCacheIndex
       ├─ GroupAllocator / BlockTable
       ├─ Device + Host BlockPool
       └─ ExecutionPlan(block_tables, pages_to_zero, L2 ops)
                    │
                    ▼
Python runtime
  ├─ generic block tables
  ├─ group block → kernel page mapping
  ├─ attention/state cache read/write
  └─ L2 / PD transfer
```

## 1. 术语

### 1.1 block 与 page

`CacheBlock` 是所有 group 通用的可寻址存储单位。只有内容由 paged KV kernel 读取时
才称 page；snapshot state block 没有 rows，也没有 page table。

- `block_table`：scheduler/runtime 搬运的通用 per-group table；
- `page_table`：KV attention kernel 消费的专用 table；
- state backend 使用 block table/state slot，不应把它命名为 page table。

### 1.2 三种 token 粒度

- `prefix_granularity`：prefix hash/match 的 identity grain；
- `CacheGroupSpec.block_granularity`：该组一个 table slot 覆盖的 raw token；
- `kernel_page_size`：attention kernel 的 page span。

三个值有各自来源。`block_granularity` 必须正向整除 P；kernel page 由
`attention/kernel_page_sizes.py` 或显式 config 决定。

### 1.3 `CacheGroupSpec`

Python spec 有两种互斥 declaration shape：

```text
row geometry        rows_per_page × entry_stride_tokens
checkpoint geometry checkpoint_granularity
```

两者都折成通用 `block_granularity`。spec 还携带 `retention`、`family`、sliding
window 与 PD transfer policy。物理 packing/page count 在 `CacheMemoryPlan`，不在 spec。

## 2. Python 物理存储

## 2.1 `CacheArena`

[`kv_cache/arena.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)
是唯一 owner：

- 分配连续 `uint8` arena；
- eager materialize memory plan 中全部 typed/strided fields；
- 发布 `CacheRuntimeContract`；
- 提供 group-aware block 清零；
- 向 PD 与 L2 暴露 plan/owner buffer。

一个合并 target+draft 模型只有一个 arena。每个 `CachePool` 是局部 layer window，
只提供 kernel-facing K/V/latent/state accessor。

## 2.2 Pool family

| family | Pool | 主要字段 |
| --- | --- | --- |
| MHA | `MHATokenToKVPool` | K/V，可选 MXFP8 scale |
| MLA | `MLATokenToKVPool` | latent，或 latent/scale/rope |
| DSA | `DSATokenToKVPool` | MLA + block-split index-K |
| MSA | `MSATokenToKVPool` | MHA + sparse index-K |
| [Qwen GDN](qwen35-lcm-cache.md) | `HybridMHATokenToKVPool` | MHA history + conv/ssm |
| Inkling | `HybridInklingTokenToKVPool` | MHA + ShortConv checkpoints |
| Kimi K3 | `HybridKDATokenToKVPool` | MLA history + KDA state |
| DeepSeek V4 | `HybridDeepseekV4TokenToKVPool` | SWA/compressed/state/indexer |

详细说明见 [CachePool 子类](cache/pools/README.md)。

## 2.3 `ReqToTokenPool`

[`cache/req_to_token_pool.py`](../python/tokenspeed/runtime/cache/req_to_token_pool.py)
保留 request slot → token row 的 runtime lookup，用于 forward 输入组织；它不是物理
CacheBlock allocator。物理 block 生命周期与 prefix reuse 都由 C++ scheduler 所有，
不能再把 `ReqToTokenPool` 与已删除的 Python `KVAllocator` 组合解释。

## 3. Python → C++ contract

[`engine/scheduler_utils.py`](../python/tokenspeed/runtime/engine/scheduler_utils.py)
只从 `pool.arena.runtime_contract` 投影：

- `scheduler_cache_geometry_from_pool()`：P、parent 总数/可用数、token capacity；
- `pool_to_cache_groups()`：每组逻辑 spec、page count、packing 与 transfer policy；
- `make_config()`：构造 `SchedulerConfig`。

snapshot state 在这一个 bridge 点折成 `(rows=checkpoint_granularity, stride=1)`，C++
只看到通用 `BlockGranularity()`。C++ 类型名是 `CacheGroupConfig`，不是旧的
`PagedCacheGroupConfig`。

`aligned_max_scheduled_tokens()` 会把 chunked-prefill budget 向下对齐到所有
full-history state group block span 的 LCM，保证 state checkpoint 恰好落在可注册
边界。

## 4. C++ cache 子系统

目录：[`tokenspeed-scheduler/csrc/cache/`](../tokenspeed-scheduler/csrc/cache/)

### 4.1 `cache/core/`

- `CacheBlockRef`：引用一个物理 block；
- `BlockTable`：per-request、per-group 的 block-id table；
- `BlockPool`：Device/Host tier 的 parent/block 资源；
- `CacheGroupConfig`：Python bridge 传入的边界配置；
- `AcquirePlan`：token-free 的分配执行计划。

null block 0 保留，allocator 的 usable parent 数为 total−1。

### 4.2 `cache/prefix/`

- `PrefixCacheIndex`：CacheKey → canonical CacheBlock 的 pool-scoped 索引；
- `FullAttnMatcher`：从左到右匹配 prefix-closed history；
- `SwaMatcher`：从右到左找可恢复 window boundary；Mamba/state snapshot 复用同一
  policy 形状；
- `prefix_hasher.h`：prefix-grain SHA-256 hashing。

prefix 模块决定「什么可复用」，不分配物理位置。

### 4.3 `cache/allocator/`

`GroupAllocator` 是唯一 allocator，没有 history/state 子类。它只处理 block 数、
placement、tier movement 和 retention 执行，不感知 token。所有 token → block count
转换由 coordinator 的 `GroupGeometry` 完成。

### 4.4 `CacheGroup`

一个 group 组合：

```text
CacheGroupSpec
+ GroupAllocator
+ PrefixMatcher
+ PrefixCacheIndex
```

这是 prefix concern 与 allocation concern 唯一交汇处。

### 4.5 `CacheCoordinator`

[`cache/coordinator/`](../tokenspeed-scheduler/csrc/cache/coordinator/) 是 scheduler
访问 cache 的唯一 facade：

- `ProbePrefix`：只读检查 Device/Host 命中并跨 group 收敛；
- `Admit`：分配、pin hit，产生 fresh blocks 与 H2D load pairs；
- `CacheFullBlocks` / `CacheCompletedBlocks`：发布可复用 prefix；
- `ReclaimExpired` / `Free` / clear：生命周期；
- 计算 admission、retraction 和 group capacity；
- 报告 cache mutation/KV events。

多个 group 共享一个物理 `BlockPool`。Coordinator 本身不保存 per-request 状态，状态
随 scheduler request/FSM 与 block tables 传递。

## 5. Prefix matching 与 admission

一次请求先 probe，后 admit：

1. 各 group 按自己的 matcher 查命中；
2. prefix-closed group 先决定公共边界，SWA/state 等 non-closed group 受该边界约束；
3. admission planner 在 shadow occupancy 上尝试使用空洞/空 parent；
4. 必要时按 LRU/tier/boundary 优先级选择最小 eviction 集；
5. 真正 admit 时 pin hit、分配缺失 block，并输出 fresh block ids。

probe 与 mutation 分离使 `CanAdmitAfterReleasing` 能在假设释放资源的情况下排序
retraction victim，而不先改动真实 cache。

## 6. Forward block table

Scheduler 发出的 table 行 i 逻辑上覆盖：

```text
[i × block_granularity, (i + 1) × block_granularity)
```

entry value 是物理 `CacheBlock` id。Python backend 只在指定 mapping 点把 generic
group table 转成 kernel page table；packing 已在 scheduler allocator/plan bridge 中
折叠，forward mapping 不再处理 `cache_blocks_per_lcm_block`。

不同 backend 的 mapping owner 当前仍分布在 MLA metadata、MHA cache-group mixin、
draft staging 和 V4 专用 metadata；它们复用公共 page-table primitive，但尚未合并成
单一 owner。

## 7. 新 block 清零

`ExecutionPlan.pages_to_zero` 是：

```text
{group_id: [CacheBlock id, ...]}
```

event loop 在 forward 前调用 `ModelExecutor.zero_cache_pages()`。只有
`requires_page_zeroing=True` 的 target/draft pool 执行 `zero_new_blocks()`；arena 根据
plan 清零该 group 的所有 field payload。清零完成 event 与 forward stream 同步，避免
旧 recurrent/live-tail 污染新请求。

## 8. Device/Host 两级 cache

C++ coordinator 同时管理 Device 与 Host `BlockPool`，`cache/tier/TransferManager`
输出 `LoadBackOp` / `WriteBackOp`。Python
[`cache/l2/executor.py`](../python/tokenspeed/runtime/cache/l2/executor.py) 消费 plan：

- `CachePool.cache_transfer_layout()` 从 memory plan 选择 layer fields；
- `cache/l2/storage.py` 分配 host LCM storage；
- executor 提交异步 D2H/H2D 并将完成事件通过 `scheduler.advance()` 回灌；
- layer-wise load tracker 在 getter 前等待相应层。

Decode role 的 Host cache 主要用于 best-effort retraction/recovery，不持续流式保存所有
普通 Device cache；Prefill/Fused 可根据配置 stream 到 Host。

## 9. L3 与 KV events

`SchedulerConfig.enable_l3_storage` 与 `enable_kv_cache_events` 控制外部 storage/KV
event 集成。C++ scheduler 可排出 `BlockStored` / `BlockRemoved` 事件；具体外部存储
backend 与请求控制面位于 runtime 配置和 event loop。不要把 L3 当成另一套 Python
radix allocator：prefix identity 和 block ownership 仍以 scheduler cache 为准。

## 10. PD 分离

PD 使用同一 plan/spec：

- `CacheTransferContract` 传 memory plan、group specs、transfer schema；
- `CachePDBlockManifest` 传请求的 per-group block ids；
- `full_suffix` 传 history/rolling state，`latest_snapshot` 只传末态 snapshot；
- source/destination TP 不同时由 `CacheTransferPlanner` 生成 field row fragments。

详见 [PD 分离](pd-disaggregation.md)。

## 11. 读代码顺序

1. [`docs/design/cache-concepts.md`](../docs/design/cache-concepts.md)
2. [`recipes/spec.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py)
3. [`recipes/plan.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)
4. [`kv_cache/arena.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)
5. [`engine/scheduler_utils.py`](../python/tokenspeed/runtime/engine/scheduler_utils.py)
6. [`cache/coordinator/cache_coordinator.cpp`](../tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.cpp)
7. 对应 attention backend 的 table mapping 与 pool accessor
