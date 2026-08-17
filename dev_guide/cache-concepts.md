# 缓存子系统深度解析：从前缀身份到物理存储

本文是 [Cache Concepts: Prefix Matching vs. Storage](../docs/design/cache-concepts.md) 的中文扩展版。它不是逐句翻译，而是沿着当前实现解释几个更重要的问题：一个 token 范围如何变成可复用前缀、多个 cache group 如何共享同一批物理内存、scheduler 发出的 block table 如何变成 kernel page table，以及一次请求在 probe、admit、publish、reclaim 和 free 之间如何维持一致性。

如果只想快速了解当前生产路径，可以先读 [KV Cache 管理机制](https://github.com/lightseekorg/tokenspeed/blob/main/dev_guide/kvcache-management.md)。本文更关注设计边界、推导过程和扩展约束。

## 1. 先建立一个正确的心智模型

tokenspeed 的 cache 子系统不是“一个按 token 编号的 KV 数组”，而是：

```text
多个逻辑 cache group
  × 每组独立的 token 几何、保留策略和前缀匹配策略
  × Device / Host 两个存储层级
  × 一个共享的 LCM 物理 parent 池
  × Python runtime 到 attention kernel 的一次表转换
```

一个模型可能同时包含 full attention、sliding-window attention、recurrent state、压缩历史和卷积状态。它们保存的数据形状不同、恢复条件不同、生命周期也不同，但 scheduler 必须把它们作为同一请求的一组约束共同调度。

理解整个设计时，应始终守住四条不变量：

1. **前缀身份在 token 空间定义，存储位置在 block 空间定义。** 两者只能通过明确的映射相交。
2. **`block` 是通用概念，`page` 是 paged KV consumer 对 block 的一种解释。** snapshot state block 不是 page。
3. **每个 cache group 独立寻址，但所有 group 动态竞争同一个物理 parent 池。** group 的可寻址容量不是静态预留容量。
4. **同一事实只能有一个来源。** prefix、group、kernel、packing 和 dtype 几何不能互相“借值”。

## 2. 两套坐标系与五个关键量

### 2.1 逻辑坐标系

逻辑坐标系只讨论 raw token 范围，不讨论字节、显存地址或 LCM packing。

| 量 | 符号 | 单位 | 含义 | 主要来源 |
| --- | --- | --- | --- | --- |
| `prefix_granularity` | `P` | token | prefix hash 和跨 group 复用的身份粒度 | cache recipe / memory plan |
| `block_granularity` | `q_g` | token | group `g` 的一个 block-table slot 覆盖多少 token | `CacheGroupSpec` |
| `page_size` | — | token | row-geometry group 的 `rows_per_page × entry_stride_tokens` | Python group spec |
| `checkpoint_granularity` | — | token | 两个 snapshot state checkpoint 之间的 token 间隔 | Python group spec |
| `kernel_page_size` | `k` | token | attention kernel 一页覆盖多少 token | kernel registry 或显式配置 |

前四个名字中，所有 `*_granularity` 以及 `page_size` 都以逻辑 token 计量，绝不能拿来表示 block 数、row 数或 byte 数。

对任意 group `g`，scheduler 要求：

```text
q_g > 0
P % q_g == 0
```

因此一个 prefix hash 可以展开成该 group 的：

```text
blocks_per_prefix_hash[g] = P / q_g
```

如果 paged KV backend 还需要把一个 scheduler block 拆成更小的 kernel page，则标准转换路径还要求：

```text
q_g % k == 0
kernel_pages_per_block[g] = q_g / k
```

这两组整除关系服务于完全不同的边界：`P % q_g == 0` 让多组 prefix identity 对齐，`q_g % k == 0` 让一个通用 block table 可被 attention kernel 消费。

### 2.2 两种互斥的 group 声明形状

Python [`CacheGroupSpec`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py) 支持两种声明形状。

第一种是 row geometry：

```text
page_size = rows_per_page × entry_stride_tokens
block_granularity = page_size
```

它适用于内容确实由若干 row 组成的 group，例如逐 token KV、压缩 KV row、SWA tail。`entry_stride_tokens > 1` 表示一行可以代表多个 raw token。

第二种是 checkpoint geometry：

```text
block_granularity = checkpoint_granularity
```

它适用于 recurrent/conv 等 snapshot state。一个 block 保存的是某个边界上的一份状态，而不是 `checkpoint_granularity` 行状态。此时 `page_size` 没有定义；调用该属性会直接抛出 `TypeError`。

两种形状互斥。判断标准是存储形状，而不是抽象的 attention family：state-family 数据如果真的包含 row，例如 DeepSeek V4 的 sliding tail 或 compressor buffer，仍应声明 row geometry。

### 2.3 物理坐标系

物理坐标系关心的是：一个 payload 放在哪个 parent、parent 内哪个 child slot、占多少字节、还有多少引用。

这里有两个容易混淆的层级：

- **LCM block / parent**：不同 group 共享的等宽物理分配单位；
- **`CacheBlock` / child block**：某个 parent 绑定到 group 后，在该 group 解释下可独立寻址的 child slot。

对 group `g`，定义：

```text
K_g = cache_blocks_per_lcm_block[g]
```

表示一个 parent 绑定到 `g` 后可容纳 `K_g` 个 child `CacheBlock`。`K_g` 是物理 packing，不是 token 粒度。

### 2.4 为什么不能用一个“block size”概括全部概念

假设：

```text
P = 128 tokens
full group:  q_full = 128 tokens, K_full = 8
SWA group:   q_swa  = 64 tokens,  K_swa  = 4
state group: q_state = 128 tokens, K_state = 1
kernel:      k = 32 tokens
```

那么：

- 一个 prefix hash 对应 1 个 full block、2 个 SWA block、1 个 state checkpoint block；
- 一个 full scheduler block 会展开成 4 个 kernel pages；
- 一个 SWA scheduler block 会展开成 2 个 kernel pages；
- 一个物理 parent 可以解释成 8 个 full child、4 个 SWA child 或 1 个 state child；
- parent 一旦有 live child，就只能采用其中一种 group 解释。

`P`、`q_g`、`k` 和 `K_g` 分别回答“身份多大”“slot 多大”“kernel 页多大”“一个 parent 塞几个 child”。把它们都叫 block size，代码迟早会在一个边界上使用错误的单位。

## 3. `block`、`page` 与两种 table

### 3.1 `CacheBlock` 是存储，page 是 consumer view

`CacheBlock` 本身不知道它保存的是 K/V row、compressed entry 还是 recurrent state。它只持有一个稳定位置：

```text
CacheBlockLocation(lcm_block_id, slot_index)
```

consumer 决定如何读取其中的字节：

- paged attention 把它看作 cache page；
- state backend 把它看作 state slot；
- allocator 只把它看作 parent 中的 occupied child。

因此每个 page 都是 block，但并非每个 block 都是 page。

### 3.2 `block_table` 与 `page_table` 不是同义词

Scheduler 维护的 [`BlockTable`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/core/block_table.h) 是通用 per-request、per-group 容器。逻辑第 `i` 个 slot 覆盖：

```text
[i × q_g, (i + 1) × q_g)
```

slot value 是物理 `CacheBlock` id，0 表示 null/hole。

`page_table` 只应出现在 paged KV kernel 的输入边界。state backend 没有 page，也不应把自己的 state block table 命名为 page table。

可以把两者的关系概括为：

```text
scheduler block_table
  ├─ paged KV consumer ──映射──> kernel page_table
  └─ state consumer ───────────> state block/slot 索引
```

## 4. 总体分层：谁可以知道什么

```text
模型层与 attention 配置
        │
        ▼
CacheRecipe: layers → group → pack → bind
        │
        ├─ CacheGroupSpec[]       逻辑策略与 group 几何
        └─ CacheMemoryPlan        parent/plane/field 字节几何
        │
        ▼
CacheArena                     唯一物理分配与 contract 发布者
        │
        ▼
pool_to_cache_groups()         Python → C++ 的唯一声明折叠点
        │
        ▼
CacheCoordinator
  ├─ PrefixMatcher/Index       什么可以复用
  ├─ GroupGeometry             token → block count
  ├─ GroupAllocator            block placement
  └─ BlockPool                 LCM parent occupancy
        │
        ▼
generic block_tables + new_page_ids + tier transfer ops
        │
        ├─ state backend
        └─ block_granularity → kernel_page_size 映射
                    │
                    ▼
             attention kernel
```

每层允许感知的信息如下：

| 层 | 可以感知 | 不应感知 |
| --- | --- | --- |
| Recipe / planner | token 几何、field shape、dtype、packing、bytes | 运行时 request ownership |
| Prefix matcher | `P`、`q_g`、window、cache key | parent placement、field bytes |
| `GroupGeometry` | token demand、`q_g`、retention | field layout、kernel page |
| `GroupAllocator` | block count、parent/slot、引用 | token、window、prefix grain |
| Scheduler/FSM | token demand、通用 block table、不透明容量视图 | row geometry、kernel page、字节 offset |
| Python mapping | group block id、`q_g`、`k` | prefix 匹配策略、LCM eviction |
| Kernel | kernel page table、field tensor view | prefix identity、LCM packing |

最强的代码审查信号是：如果 `tokenspeed-scheduler` 中出现 `page_size`，说明 kernel/KV row 概念泄漏进了通用 scheduler；如果 allocator 开始接收 token 或 window，则逻辑策略泄漏进了 placement。

## 5. Python 侧：从模型声明到一块 arena

### 5.1 `layers → group → pack → bind`

所有模型 family 都经由 [`CacheRecipe.setup()`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/recipes/base.py) 执行相同流水线：

```text
layers ──group──> groups ──pack──> CacheLayout ──bind(N)──> CacheMemoryPlan
```

四个阶段分别回答：

1. **layers**：模型有哪些 target/draft layer，每层属于什么 attention 类型和 group；
2. **group**：哪些 layer 共享一个 group，每个 group 的逻辑 spec 和 fields 是什么；
3. **pack**：一个 capacity-independent parent 内，各 plane、field offset、page stride 和 `K_g` 是什么；
4. **bind**：显存预算最终能提供多少个 parent，并据此生成完整 plan。

`group()` 一次遍历同时生成 `(CacheGroupSpec, fields)`。group id 只在 spec 旁边声明一次，field 不再携带另一份可能冲突的 group id；`pack()` 直接消费这些 pair。因此“spec 中的 group 集合”和“plan 中的 group 集合”按构造方式天然一致，不需要事后 reconcile。

### 5.2 `pack()` 求解的是一个 parent，不是总容量

[`pack()`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py) 先按 plane 聚合 fields，再根据 payload 字节比例、exact-stride 约束、dtype element size、alignment 和 padding 上限求出每组 `K_g`。

对 plane `p`，可用下面的简化式理解它的宽度：

```text
plane_bytes[p]
  = aligned_max_g(K_g × bytes_of_group_g_on_plane_p)
```

一个 parent 的总字节数为：

```text
L = align(sum_p plane_bytes[p])
```

`CacheLayout` 到这里仍与显存容量无关，只描述一个 parent。只有 `bind(N)` 才引入 parent 数量 `N`。

### 5.3 group overlay 与 parent 独占绑定

不同 group 的 field 可以 overlay 同一组 plane 字节，但 overlay 不表示它们可以同时占用同一 parent。一个 parent 的运行时状态是：

```text
free
  或 bound(group=A, occupancy=[...])
  或 bound(group=B, occupancy=[...])
```

只要任意 child 仍被 request table、prefix index 或异步 transfer ticket 引用，parent 就保持原 group binding。最后一个 child 释放后，parent 才清除 binding 并回到 free FIFO。

[`BlockPool`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/core/block_pool.h) 分配时优先填充同 group 已部分占用且 occupancy 较高的 parent，再消费 free parent。这降低了内部碎片，也避免为了更紧凑的 packing 主动驱逐仍可用的 cache。

### 5.4 `bind(N)` 后的容量与 null parent

设 `N` 为可用 LCM parent 数，不包含 null parent，则：

```text
arena_bytes = (N + 1) × L
page_count[g] = 1 + N × K_g
```

前面的 `+1` 对应 id 0 的 null block。arena 会为 null parent 真实保留一份零字节区域，使 state 的 zero input、block-table hole 和拆分后的 kernel null pages 都有合法落点。

`CacheRuntimeContract` 会验证：

```text
group_page_counts[g] == N × K_g + 1
```

因此 scheduler 收到的 group count 和 packing 不是另一套估算，而是 memory plan 的直接投影。

### 5.5 child block id 与字节地址

parent/slot 到对外 child id 的映射由 [`GroupAllocator::ResolveCacheBlockId`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/allocator/group_allocator.h) 唯一定义：

```text
child_id = 1 + (parent_id - 1) × K_g + slot_index
```

例如 `K_full = 8` 时，parent 3 的 child id 是 17 到 24；若同一 parent 改为绑定 `K_state = 1` 的 state group，它的 child id 是 3。一个整数 id 必须始终与 group id 一起解释。

字段地址则由 `CacheMemoryPlan.field_page_byte_offset()` 统一计算。抽象地看：

```text
field address
  = plane arena base
  + parent-local stride derived from child_id
  + field offset in that plane
```

forward、清零、Host L2 和 PD 都应复用 plan 的地址计算，不能分别复制 `(parent, slot)` 公式。

### 5.6 `CacheArena` 为什么是唯一 owner

[`CacheArena`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/arena.py) 一次性完成：

- 分配连续 `uint8` buffer；
- 按 memory plan eager materialize 所有 typed/strided field view；
- 保存 plan 和几何 scalar；
- 发布 scheduler-facing `CacheRuntimeContract`；
- 按 group 和 field byte range 清零新 block；
- 向 L2/PD 暴露同一 owner buffer。

`CachePool` 只是 arena 上的 layer window 和 kernel-facing typed view，不拥有另一份内存，也不应镜像 plan、prefix grain 或 packing。target 与 draft 合并时仍只有一个 arena 和一个 contract，从根本上避免两套 page-id 空间漂移。

## 6. Python → C++：声明形状在这里折叠

同一个 group 穿过三种类型：

```text
Python CacheGroupSpec
  declaration shape + policy
        │
        ▼ pool_to_cache_groups()
C++ CacheGroupConfig
  nanobind 边界配置
        │
        ▼ MakeSpecsFromConfig
C++ CacheGroupSpec
  block_granularity + kind + packing
```

[`pool_to_cache_groups()`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/engine/scheduler_utils.py) 是唯一折叠点：

```text
row geometry:
  rows_per_page, entry_stride_tokens 原样过桥

snapshot geometry:
  rows_per_page = checkpoint_granularity
  entry_stride_tokens = 1
```

C++ 统一计算：

```text
BlockGranularity() = rows_per_page × entry_stride_tokens
```

这不是在声称 snapshot 真有 rows，而是边界协议用两个整数编码通用 token span。`checkpoint_granularity` 这个声明词到 bridge 为止；C++ scheduler 只讨论一个 table slot 覆盖多少 token。

配置验证至少应保证：

- group id 非空且唯一；
- rows/stride/total count/packing 为正；
- sliding group 的 window 为正；
- `q_g` 是 `P` 的正因子；
- snapshot state、row geometry 和 transfer policy 的组合合法；
- recurrent state 的调度 chunk 能落在可物化 checkpoint 边界。

## 7. C++ cache 核心对象与所有权

### 7.1 `BlockPool`：只管理 placement

`BlockPool` 保存每个 parent 的：

```text
bound_group
occupancy bitmap
occupied_count
```

它没有 cache key、LRU、token、window 或 field shape。一个空 parent 第一次分配 child 时绑定 group；最后一个 child 被释放时解绑。

### 7.2 `CacheBlockRef`：引用就是 pin

[`CacheBlockRef`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/core/cache_block_ref.h) 是 pool-scoped shared owner。常见 owner 包括：

- 当前请求的 `BlockTable`；
- `PrefixCacheIndex` 中的可复用 cache entry；
- in-flight Host/Device transfer ticket；
- admission 期间刚 acquire 的临时结果。

最后一个引用释放时，embedded `CacheBlock` 析构并通知 `BlockPool::Release(location)`。所以“cache entry 是否可淘汰”不是一个孤立布尔值：只有 index 自己是唯一 owner，或者 planner 明确知道哪些 owner 即将释放，location 才真正可回收。

### 7.3 `BlockTable`：逻辑长度不一定等于 live block 数

`BlockTable` 保存 `vector<CacheBlockRef>` 和尾部尚未消费的 `available_tokens`。SWA/state reclaim 不会缩短 vector，而是把过期前缀替换为 null holes：

```text
[block0, block1, block2, block3]
        │ reclaim first 2
        ▼
[null, null, block2, block3]
```

保持 slot 位置稳定很重要：prefix key 的 `page_offset`、远端 snapshot 的 sparse suffix 和 runtime table 对齐都依赖逻辑 index 不随 reclaim 左移。

### 7.4 `PrefixCacheIndex`：身份到 canonical block

[`PrefixCacheIndex`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/prefix/prefix_index.h) 维护：

```text
CacheKey → canonical CacheBlockRef
location → CacheEntry
```

一个 index 对应一个 group，但内部按 `BlockPool*` 区分 Device/Host tier，因此同一个 group index 可以服务两个 pool。

当 `Register()` 发现相同 key 已有 canonical block 时，新 block ref 会被替换为 canonical ref。新分配但内容重复的物理 child 随最后一个引用释放，不会在索引中留下两份等价数据。

### 7.5 `CacheGroup`：两类 concern 的唯一交汇点

```text
CacheGroup
  = CacheGroupSpec
  + GroupAllocator
  + PrefixMatcher
  + PrefixCacheIndex
```

matcher 只决定什么可复用，allocator 只决定放在哪里；`CacheGroup` 把同一 group 的两者组合起来，但不把它们的数据结构混在一起。

### 7.6 `CacheCoordinator`：scheduler 的唯一 facade

[`CacheCoordinator`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.h) 将一次 request-level 操作 fan out 到所有 group，再收敛为一个结果。它自身不持有 per-request block tables；request/FSM 持有 tables 和 issued access epoch，coordinator 只维护全局 cache 状态与 epoch 时钟。

## 8. Prefix identity：一个 hash 如何展开成多组 key

上游按每 `P` 个 token 生成累积 content hash。对 group `g`，coordinator 将每个 hash 展开为 `P / q_g` 个 key：

```text
CacheKey {
  namespace_id,
  group_id,
  content_hash,
  page_offset
}
```

假设 `P=128`、`q_swa=32`，一个 prefix hash `H7` 会生成：

```text
(group=swa, hash=H7, offset=0)
(group=swa, hash=H7, offset=1)
(group=swa, hash=H7, offset=2)
(group=swa, hash=H7, offset=3)
```

`page_offset` 是“group block 在一个 prefix identity grain 中的序号”，不是 byte offset，也不是 token offset。这样，prefix identity 仍以 `P` 为边界，而较细 group 可以在同一个 hash 下注册多个 block。

`namespace_id` 用于隔离不应互相复用的 cache 域；`group_id` 防止不同数据形状即使 content hash 相同也误用彼此的 storage。

## 9. 两种 matcher：闭合前缀与可恢复边界

### 9.1 Full attention：prefix-closed

[`FullAttnMatcher`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/prefix/prefix_matcher.h) 从左向右扫描，遇到第一个 miss 就停止：

```text
keys:  0 1 2 3 4 5
hit:   ✓ ✓ ✓ ✗ ✓ ✓
match: ✓ ✓ ✓
```

因为 full attention 恢复位置 `b` 需要 `[0, b)` 的完整历史。若长前缀可恢复，则任何更短前缀也可恢复，所以它是 prefix-closed。

### 9.2 SWA：不是“最长连续前缀”

SWA 恢复边界 `b` 只需要该边界之前覆盖最近 `W-1` 个 token 的连续 pages：

```text
lookback_pages = ceil((W - 1) / q_g)
```

[`SwaMatcher`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/prefix/prefix_matcher.h) 从右向左寻找最高的、拥有足够连续 lookback pages 的边界。较早位置出现 hole 不一定阻止恢复：

```text
keys:      0 1 2 3 4 5 6
hit:       ✗ ✗ ✓ ✓ ✗ ✓ ✓
需要 2 页 lookback
可恢复边界:                    ^ boundary=7
```

这类 match 不是 prefix-closed。盲目把边界缩短到 6，尾部连续 run 可能只剩一页，反而不再满足恢复条件。

### 9.3 Mamba/state 为什么复用 SWA matcher

snapshot state 的恢复形状与窗口为 2 的 SWA 相同：保留 live state block 和它的前一份 snapshot。因此 C++ 不需要 `MambaMatcher` 子类，而是用 `SwaMatcher(q_g, window=2)` 表达同一种“边界前需要连续 lookback”的策略。

这只是 match policy 的复用，不表示 state 变成了 paged KV，也不表示模型 window 真的是 2 token。

### 9.4 多 group 必须收敛到同一个边界

一次 request 只有在所有 group 都能恢复时才能跳过 prefix 计算。coordinator 的匹配顺序是：

1. 先匹配 prefix-closed groups，得到完整历史可支持的上界；
2. 再让 SWA/state 等 non-closed groups 在此上界内寻找可恢复边界；
3. 如果后面的 group 又缩短了公共边界，重新匹配任何仍高于新边界的 non-closed group；
4. 最终结果向下对齐到 `P`。

重新匹配是必要的：non-closed group 在较高边界成立，不代表把它截断到另一个 group 的较低边界后仍成立。

## 10. Probe 与 Admit：先证明能放下，再改变状态

### 10.1 为什么分成两个阶段

`ProbePrefix()` 只读查询 Device/Host 命中，不 acquire 引用、不更新 LRU、不淘汰。`Admit()` 才执行 pin、evict、reclaim、Host load destination 分配和 fresh block 分配。

两者分离带来两个能力：

- scheduler 可以在“不真实释放请求”的情况下调用 `CanAdmitAfterReleasing()`，比较 retraction candidate；
- 容量不足时返回失败，不留下半完成的 cache mutation。

从 probe 到对应 admit 之间，cache 状态不得被其他路径改变。Admit 一旦通过 shadow plan 并进入 commit，内部 plan/pool 不一致被视为 fatal invariant violation，而不是尝试复杂回滚。

### 10.2 Admission planner 的 shadow occupancy

[`AdmissionPlanner`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/coordinator/cache_admission.cpp) 不修改真实 pool，而是构造：

- 每个 parent 的模拟 occupied count；
- 每个 group 已绑定 parent 中的 local free slots；
- 完全空闲 parent 数；
- 每个 group 本次仍需放置的 child block 数。

对 group `g`，若本地 hole 可先吸收 `h_g` 个 child，本次需要 `b_g` 个 child，则仍需新 parent：

```text
parents_g = ceil(max(b_g - h_g, 0) / K_g)
```

整个请求可放下的判定为：

```text
sum_g(parents_g) <= empty_parent_count
```

这个公式解释了为什么不能只看所有 group 的 page count 总和：一个 partially-filled parent 的剩余 slot 只能给当前绑定 group 使用。

### 10.3 eviction 顺序

当前 prefix hits 会先被保护，不进入 victim 集。其余候选主要按以下信息排序：

1. 较旧的 `last_access_epoch` 先淘汰；
2. 同一 epoch 内，request-only uncached block 优先；
3. 再考虑尚未证明价值的 probationary boundary；
4. 然后是 established non-closed boundary；
5. 最后才是 closed prefix，并倾向从 suffix 回收。

Planner 先按顺序 pop victim，直到模拟容量可容纳 admission；随后逆序尝试恢复 victim，只保留确实不可少的集合。结果不是简单“淘汰到够用”，而是在当前 candidate 顺序下消除冗余 eviction。

### 10.4 Admit 的提交顺序

容量计划成功后，核心提交过程是：

1. 为命中的 Device/Host cache entry acquire owning refs，并更新 access epoch；
2. 把 Device hits 安装到新的 request tables；
3. 淘汰已经可释放的 victim；
4. 发布上一阶段刚完成的 boundary，并 reclaim 当前请求已过期 slots；
5. 再处理因 request ref 刚释放而变得可淘汰的 victim；
6. 为 Host hits 分配 Device destination，生成 H2D `load_pairs`；
7. 按 group 执行 fresh allocation；
8. 输出每组 `new_page_ids`，供 forward 前清零。

顺序的关键在于引用关系：某个 block 在 planner 看来“释放 request ref 后可淘汰”，但提交早期仍可能同时被 request table 和 prefix index pin，必须等 reclaim 发生后再重试。

## 11. 一次请求的完整 cache 生命周期

| 阶段 | 读取 | 输出 | 真实 mutation |
| --- | --- | --- | --- |
| 生成 prefix hashes | 完整 `P` 边界 token | cumulative hashes | 无 |
| `ProbePrefix` | hashes、各组 index | Device/Host probe、公共边界 | 无 |
| admission planning | probe、demands、shadow occupancy | 可行性、victims | 无 |
| `Admit` | plan、request tables | hits、loads、fresh ids、epoch | pin/evict/reclaim/allocate |
| fresh block 清零 | `new_page_ids`、memory plan | zero-complete event | 写 arena |
| Host load | `load_pairs` | transfer completion | 写 Device arena |
| forward | block tables、field views | KV/state 更新 | 写 live blocks |
| publish | completed hashes/boundary | reusable entries | index acquire refs |
| reclaim | computed token count | null holes | request refs 释放 |
| request free | 全部 group tables | 空 tables | request refs 释放 |

### 11.1 为什么 fresh block 必须显式清零

parent 会跨 group 重用。旧 parent 可能曾保存 FP8 KV，下一次却按 FP32 recurrent state 解释同一批字节；不清零可能直接产生巨大状态或 NaN。

Scheduler 返回：

```text
new_page_ids[group_id] = [child block ids...]
```

Python runtime 只对 `requires_page_zeroing=True` 的 pool 执行 group-aware clear。`CacheArena.zero_blocks()` 根据 plan 找到该 group 的所有 field payload byte ranges，再调用统一 zero kernel。清零完成事件必须与 forward stream 正确同步。

### 11.2 publish 不等于 request free

一个 block publish 到 prefix index 后通常有两个 owner：

```text
request BlockTable ref + PrefixCacheIndex ref
```

request 结束时只释放前者，block 仍作为 cache 存活。之后 admission planner 淘汰 index entry，或显式 clear cache，最后一个 ref 才让 parent slot 回收。

### 11.3 closed 与 non-closed group 的发布范围不同

Full attention 可以发布所有新完成的完整 blocks。SWA/state 只发布恢复某个 boundary 所需的 trailing lookback blocks。

Mamba state 还要求 kernel 确实在相应 `P` 边界物化了 checkpoint；如果 `num_computed_tokens % P != 0`，不能把“尚不存在的边界状态”注册进 cache。普通 SWA 保存的是 KV rows，可以发布未对齐 endpoint 之前最后一个完整 page boundary。

## 12. 从 scheduler block table 到 kernel page table

Scheduler table 的行位置是逻辑的，entry value 是 opaque child id。Python 的标准转换 primitive 是 [`expand_page_table()`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/page_table.py)。

若：

```text
r = q_g / k
```

则一个 scheduler child id `b` 展开为：

```text
[b × r + 0, b × r + 1, ..., b × r + (r - 1)]
```

例如 `q_g=128`、`k=32`，table entry 17 会展开成 kernel pages `[68, 69, 70, 71]`。null block 0 对应 null parent 内的多个 zero subpages。

这个转换不再处理 `K_g`。LCM `(parent, slot) → child id` 已由 C++ allocator 和 Python plan 共同固定；forward 再碰 packing 会造成重复映射。

还要注意两点：

- state backend 直接消费 state block ids，不执行 kernel page expansion；
- 当前公共 expansion primitive 已统一，但 mapping owner 仍分布在 MHA mixin、MLA metadata、draft staging 和 DeepSeek V4 专用路径。新增路径应复用 primitive，不应再造第五套 page 算术。

## 13. Device / Host 两级 cache

Device 和 Host 各自有 `BlockPool`，但每个 group 共享同一个 `PrefixCacheIndex` 对象，index 内部按 pool 区分 entry。一次普通 probe 的顺序是：

```text
先在 Device 从 token 0 匹配
  │
  ▼
以 Device 公共边界作为 floor，再到 Host 继续匹配
```

Host 命中不会直接进入 Device block table。Admit 会：

1. pin Host source；
2. 在 Device 为每个非 hole source 分配 destination；
3. 在 table 中保留相同的逻辑 slot/hole 形状；
4. 输出 `BlockTransfer{group_id, source, destination}`；
5. runtime 执行 H2D，并把完成事件回灌 scheduler。

Device publish 是否自动排入 D2H `pending_stores_` 由 `stream_device_cache_to_host` 控制。异步 store ticket 本身会 pin source；admission 的 hypothetical-release 只能把这些“完成后将释放”的 ref 计入模拟，不能提前修改真实 pool。

Host 容量不足时，coordinator 优先淘汰同 group parent 中的一个 child，因为这样可以直接复用 local slot；只有必要时才整体淘汰另一个 group 的 fully-evictable parent 并重新绑定。

## 14. PD 分离中的 group 语义

PD 仍使用同一份 memory plan、group specs 和 child id 语义。transfer policy 由 group 的数据形状决定：

```text
history 或 sliding state  -> full_suffix
full-history snapshot state -> latest_snapshot
```

Decode 侧恢复 snapshot state 时，可以在早期逻辑 slots 中保留 null holes，只物化远端 endpoint 附近的 suffix。`GroupDemand.materialized_suffix_start` 和 `AcquirePlan.suffix_start` 明确表达这个 sparse table：allocator 只按给定 block count 和 slot 起点执行，不需要知道为什么这些 hole 合法。

`ProbeDecodeDevicePrefix()` 也遵循这一语义：本地 history group 可以复用完整前缀，而 final-state groups 留出对齐 hole，等待远端 latest snapshot 填充。hole 不算 cache hit。

## 15. 容量、碎片与为什么 token capacity 只是一个视图

### 15.1 每个 group 的地址空间

绑定 `N` 个 parent 后，group `g` 的可用 child 数是：

```text
usable_child_blocks[g] = N × K_g
```

但这些地址不是静态占据的物理分区。多个 group 各自拥有完整的 id 公式，却在运行时竞争同一批 `N` 个 parent。

### 15.2 一个混合请求需要多少 parent

忽略 prefix hit 和 partial parent 时，下界是：

```text
parents_needed = sum_g ceil(child_demand[g] / K_g)
```

考虑已绑定 parent 的 local holes 后，应使用 admission planner 的公式：

```text
parents_needed
  = sum_g ceil(max(child_demand[g] - local_free_slots[g], 0) / K_g)
```

因此：

- `N × max(K_g) × P` 只能描述最细 packing group 的扁平 token 视图；
- state group 可能按“每请求几个 checkpoint block”消耗，而非按 token 线性消耗；
- SWA group 的 live demand 受 window 限制；
- parent 的 group-exclusive binding 会产生暂时无法跨组利用的碎片。

诊断 admission failure 时，应同时看 group demand、`K_g`、partial parent occupancy、pinned cache 和 pending transfer，而不是只看总 token capacity。

### 15.3 一个数值例子

假设：

```text
N = 10 parents
K_full = 8
K_swa = 4
K_state = 1
```

某轮需要新放置：

```text
full: 17 children
SWA:   5 children
state: 2 children
```

没有 local holes 时：

```text
ceil(17 / 8) + ceil(5 / 4) + ceil(2 / 1)
= 3 + 2 + 2
= 7 parents
```

即使 full group 的地址空间有 80 个 child ids，也不表示另外两个 group 已各自获得 10 个 parent。若当前已有 4 个 parent 被其他 group 的少量 pinned child 绑定，剩余 6 个空 parent 就不足以满足本次 7-parent admission。

## 16. 并发与一致性边界

Cache 正确性不仅依赖几何，还依赖几个时序约束。

### 16.1 access epoch 是 request 生命周期属性

新请求第一次 admit 时由 coordinator 发放 epoch；同一请求后续轮次携带原 epoch。prefix entry 更新 `last_access_epoch`，admission 用它比较 cache value。Coordinator 不存 per-request 状态，因此 request/FSM 必须正确携带 epoch。

### 16.2 probe 结果是只读快照，不是长期 lease

Probe 没有 pin block。如果 probe 后允许无关 mutation 改变 index，再拿旧 probe admit，就可能出现“命中在 acquire 前消失”。当前设计要求 probe 结果紧接着交给对应 admit，内部通过 assertion 把违反此约束视为逻辑错误。

### 16.3 异步 transfer 通过引用维持存活

D2H/H2D 操作必须持有 source/destination refs，直到完成事件被确认。不能因为 request table 已 reclaim 就假定底层位置可重用。`pending_store_releases` 只用于容量预测；真实 ref 仍由 ticket 生命周期释放。

### 16.4 清零与 forward 必须有显式依赖

`new_page_ids` 是“本次新获得且内容未定义”的集合。event loop 必须保证 zero/H2D 完成后 kernel 才能读取相应 blocks；依赖 Python 执行顺序而没有 stream event，会在 overlap schedule 下产生竞态。

## 17. 新增 cache group 或 backend 的设计步骤

### 17.1 先声明 consumer 的数据形状

依次回答：

1. 数据是按 row 保存，还是每隔若干 token 保存一份 snapshot？
2. 它属于 history 还是 state family？
3. retention 是 full history 还是 sliding window？
4. PD 需要 full suffix 还是 latest snapshot？
5. 一个逻辑 block 覆盖多少 raw token，即 `q_g` 是多少？

不要先从已有 `page_size` 复制一个数字，再倒推这些语义。

### 17.2 在 recipe 中一次声明 spec 和 bytes

- 为 group 生成唯一 id；
- 把 spec 与 fields 作为 pair 交给 `pack()`；
- field dtype、shape、plane、alignment 和 exact stride 在 plan 层明确；
- packing 由 byte ratio 求解，只有模型确有固定布局策略时才显式 pin；
- 用 `check_layout()` 校验 kernel 真正要求的几何。

### 17.3 保持三个粒度的来源独立

- `P` 来自 cache recipe / memory plan；
- `q_g` 来自 group spec；
- `k` 来自 kernel registry 或显式 config。

如果需要整除关系，直接校验关系并给出诊断；不要把一个值作为另一个值的 fallback。

### 17.4 选择已有 matcher 形状

- 需要 `[0,b)` 全历史：`FullAttnMatcher`；
- 只需要 boundary 前固定 lookback：`SwaMatcher`；
- 只有现有两种恢复数学都无法表达时，才新增 matcher。

Matcher 不得 acquire storage 或操作 parent。

### 17.5 runtime 只增加必要的 consumer view

- paged KV backend 复用公共 page expansion；
- state backend 使用 block/state slot，不引入 page table；
- 字节地址从 plan 查询；
- packing 不进入 forward；
- fresh bytes 若可能包含旧数据，声明并测试 group-aware zeroing。

## 18. 常见错误及其根因

| 错误 | 根因 | 正确做法 |
| --- | --- | --- |
| 从 `prefix_granularity` 推导 kernel page | 混淆身份与 kernel 几何 | 从 registry/config 取 `k`，显式校验整除 |
| 在 scheduler 中传 `page_size` | paged KV view 泄漏 | 使用通用 `block_granularity` |
| 给 snapshot state 声明虚构 rows | 把编码形式当数据形状 | 使用 `checkpoint_granularity`，只在 bridge 折叠 |
| 把 block-table entry 当 parent id | 忽略 group packing | entry 是 child id；parent/slot 只在 allocator/plan 内解释 |
| 认为各 group 有独立物理 pool | 把 id 空间当静态容量 | 所有 group 动态竞争共享 LCM parents |
| reclaim 时删除 table 前缀 | 破坏逻辑 slot 对齐 | 把 expired slots 置为 null holes |
| forward 再计算 `K_g` | 重复物理映射 | C++ 已输出 child id，Python 只做 `q_g → k` |
| 只命中 full KV 就跳过 prefill | 忽略 state/SWA 恢复条件 | 所有 group 收敛到公共 boundary |
| 新 block 不清零 | parent 跨 group/dtype 重用 | 根据 plan 清零该 group 全部 field payload |
| 为 allocator 增加 SWA/Mamba 子类 | retention policy 泄漏到 placement | `GroupGeometry` 计算 expired/acquire plan，allocator 只执行 |
| 用总 free child 数判断 admission | 忽略 parent binding 碎片 | 按 group local holes 与 `K_g` 在 parent 层规划 |

## 19. 排障方法

### 19.1 几何或启动校验失败

按来源从上到下检查：

1. recipe 发布的 `P`、group ids 和 group specs；
2. 每组声明形状是否互斥且 `P % q_g == 0`；
3. paged consumer 是否满足 `q_g % k == 0`；
4. `CacheLayout.group_packing`、plane bytes、field strides；
5. `CacheMemoryPlan.page_count == N × K_g + 1`；
6. runtime contract 是否与 plan 使用同一组 ids。

### 19.2 Prefix 命中比预期短

检查：

- content hashes 是否只覆盖完整 `P` 边界；
- `CacheKey.namespace_id/group_id/page_offset` 是否一致；
- full group 是否在某个更早位置 miss；
- SWA/state 在最终公共 boundary 前是否有足够连续 lookback；
- non-closed group 是否因另一个 group 缩短边界而重新匹配；
- 目标数据在 Device、Host 还是仍处于 pending store。

### 19.3 明明有空闲 child id 却 admission 失败

重点看 parent，而不是 child id 上限：

- 空 parent 数；
- 每个 parent 的 bound group 与 occupied count；
- 需要的 group 是否有 local holes；
- sibling child 是否被 request/index/transfer pin；
- victim 是否因当前 prefix hit 被保护；
- hypothetical release 是否真的覆盖 parent 内全部 live children。

### 19.4 Forward 出现 NaN、错位或跨请求污染

优先排查：

- `new_page_ids` 是否按正确 group 进入 zero path；
- zero/H2D 与 forward stream 是否同步；
- state backend 是否误用了 history table；
- block id 是否在错误 group 作用域解释；
- kernel page expansion 是否使用了错误的 `q_g` 或 `k`；
- field dtype/stride 是否来自 plan，而不是 backend 重算。

## 20. 测试与代码审查清单

新增或修改 cache 行为时，至少覆盖：

- spec shape 的合法/非法组合；
- `P`、`q_g`、`k` 的整除边界；
- 一个 prefix hash 展开多个 group keys 的 `page_offset`；
- full matcher first-miss 与 SWA matcher right-to-left boundary；
- 多个 non-closed groups 的反复收敛；
- parent partial-fill、group-exclusive binding 和最后引用释放；
- canonical registration 替换重复 block；
- zero-eviction admission、需要 eviction、容量不足无 mutation；
- reverse restore 后 victim 集最小化；
- reclaim 产生 null holes 而不移动 slot；
- Host extension 保留 hole 形状并输出正确 load pairs；
- fresh block 按 group 清零；
- `q_g > k` 时 page table 展开；
- snapshot state 不走 page table；
- PD sparse suffix / latest snapshot；
- target/draft 共用 arena 和 contract。

仓库中的主要测试入口包括：

- [`tokenspeed-scheduler/tests/cpp/test_cache_scenarios.cpp`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/tests/cpp/test_cache_scenarios.cpp)
- [`tokenspeed-scheduler/tests/cpp/test_cache_operations.cpp`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/tests/cpp/test_cache_operations.cpp)
- [`tokenspeed-scheduler/tests/cpp/test_cache_coordinator.cpp`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/tests/cpp/test_cache_coordinator.cpp)
- [`tokenspeed-scheduler/tests/cpp/test_cache_block_ref.cpp`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/tests/cpp/test_cache_block_ref.cpp)
- [`test/runtime/test_cache_pool.py`](https://github.com/lightseekorg/tokenspeed/blob/main/test/runtime/test_cache_pool.py)
- [`test/runtime/test_prefix_cache_e2e.py`](https://github.com/lightseekorg/tokenspeed/blob/main/test/runtime/test_prefix_cache_e2e.py)
- [`test/test_cache_transfer_layout.py`](https://github.com/lightseekorg/tokenspeed/blob/main/test/test_cache_transfer_layout.py)

## 21. 推荐读代码顺序

1. [`recipes/spec.py`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py)：逻辑 group 形状与 policy；
2. [`recipes/plan.py`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)：field、plane、packing、bind；
3. [`arena.py`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)：唯一 allocation owner 与 runtime contract；
4. [`scheduler_utils.py`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/engine/scheduler_utils.py)：Python → C++ bridge；
5. [`cache_types.h`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/core/cache_types.h) 与 [`group_geometry.h`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/coordinator/group_geometry.h)：C++ 通用 spec 和 token 算术；
6. [`prefix_matcher.h`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/prefix/prefix_matcher.h) 与 [`prefix_index.h`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/prefix/prefix_index.h)：复用策略与 canonical index；
7. [`block_pool.h`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/core/block_pool.h) 与 [`group_allocator.h`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/allocator/group_allocator.h)：parent/child placement；
8. [`cache_admission.cpp`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/coordinator/cache_admission.cpp)：shadow capacity 与 eviction；
9. [`cache_coordinator.cpp`](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.cpp)：跨 group/tier 生命周期；
10. [`page_table.py`](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/page_table.py) 和对应 backend：最后一次 block → kernel page 映射。

## 22. 最终检查：一句话能否说清每个量

在设计或评审 cache 改动时，可以用下面五句话做最后检查：

- `P` 决定“两个请求在哪些 token 边界上可以被认作同一前缀”；
- `q_g` 决定“group `g` 的 block table 一个 slot 代表多少 token”；
- `k` 决定“attention kernel 一次按多大的 token page 寻址”；
- `K_g` 决定“一个 LCM parent 绑定到 group `g` 后能切成几个 child blocks”；
- `L` 决定“一个物理 parent 实际占多少字节”。

只要这五个答案各自有唯一来源、跨边界时有显式映射，prefix matching、scheduler admission、物理复用和 kernel 执行就能保持解耦；一旦其中两个量开始共享名字或互相推导，通常就是下一次 cache 几何 bug 的起点。
