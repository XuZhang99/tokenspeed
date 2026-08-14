# CacheMemoryPlan：容量绑定的物理计划

[上一篇：CacheLayout](layout.md) · [返回目录](../cache.md) · [下一篇：CachePoolSpec](pool-spec.md)

`CachePool` 不自行计算物理布局，而是执行 recipe 生成的 `CacheMemoryPlan`。
该类型定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)。

`CacheMemoryPlan` 是容量已经确定的静态字节几何。它不持有 Tensor、dtype、device
或 scheduler 状态，只使用整数和 tuple 描述一块共享 LCM arena 应当怎样分配与解释。

```python
@dataclass(frozen=True)
class CacheMemoryPlan:
    logical_block_tokens: int
    lcm_block_bytes: int
    num_lcm_blocks: int
    groups: tuple[CacheGroupLayout, ...]
    planes: tuple[CachePlaneLayout, ...] = ()
    fields: tuple[CacheFieldLayout, ...] = ()
```

它在 cache 创建链路中的位置如下：

```text
CacheFieldSpec
    ↓ solve_cache_layout()
CacheLayout                         容量无关：描述一个 LCM 父块
    ↓ with_num_lcm_blocks(N)
CacheMemoryPlan                     容量已绑定：描述完整 arena
    ↓ 放入 CachePoolSpec
create_cache_pool()
    ↓
CachePool                           分配 arena 并建立 Tensor view
```

顶层字段的含义如下：

| 字段 | 含义 |
| --- | --- |
| `logical_block_tokens` | 一个逻辑 CacheBlock 表示的 token 数，也是 runtime contract 的基础 block size。 |
| `lcm_block_bytes` | 一个 LCM 父块跨全部 plane 的总字节数。 |
| `num_lcm_blocks` | 可用 LCM 父块数量，不包含保留的 null parent。 |
| `groups` | 每个逻辑 page-id 空间的 packing 和最终 page 数。 |
| `planes` | 完整 arena 中各物理 slab 的宽度和起始 offset。 |
| `fields` | 计算字段在 group、plane 和 child page 内的具体几何。 |

`frozen=True` 只阻止字段被重新赋值；更重要的是，生产路径通过已经校验过的
`CacheLayout.with_num_lcm_blocks()` 创建 plan，使 pool、scheduler、PD 和 Host L2
看到同一份不可变几何。

Plan 中最重要的三个集合分别处在不同层次：

```text
groups：逻辑 page-id 空间与 packing
planes：物理内存及复用关系
fields：计算代码看到的 Tensor 视图
```

除此之外，plan 还保存 `num_lcm_blocks`、`lcm_block_bytes` 和
`logical_block_tokens` 等整体几何信息。

## `logical_block_tokens`：统一的逻辑分页粒度

`logical_block_tokens` 通常记为 `P`，表示 scheduler 的一个逻辑 page 对应多少个
token。它是整个 plan 共享的逻辑分页粒度，创建 runtime contract 时会成为：

```python
PagedCacheRuntimeContract(
    block_size=plan.logical_block_tokens,
    ...,
)
```

普通 MHA/MLA 场景中，它通常等于 `AttentionConfig.page_size`。例如：

```text
logical_block_tokens = 128
```

表示 scheduler 每 128 个 token 使用一个逻辑 page。普通 MHA 的 K/V field 页内
shape 也会以它作为第一个维度：

```text
K page shape = [128, kv_head_num, head_dim]
V page shape = [128, kv_head_num, head_dim]
```

### 它参与哪些计算

它主要用于：

- 定义 scheduler/runtime contract 的基础 `block_size`；
- 计算普通 group 的几何 token 槽位数；
- 检查 backend kernel page size 能否整除 scheduler logical page；
- 协调不同 cache group 在同一个 LCM plan 下的分页粒度。

例如有 10 个可用 LCM 父块，某 group 的 packing 为 4：

```text
usable child pages = 10 × 4 = 40
page_count = 1 + 40 = 41              # 包含 null page 0
geometric token slots = 40 × 128 = 5120
```

`CachePoolSpec.pool_size` 也使用最大 group packing 计算：

```text
pool_size = num_lcm_blocks × max_packing × logical_block_tokens
```

### 与 kernel/group page size 的区别

普通 MHA 中以下三个值通常相同：

```text
logical_block_tokens
    = scheduler logical page size
    = KV field 每页的 token 行数
    = attention kernel page size
```

异构 cache 中不能假设它们始终相同。Backend 的 kernel page 可以更小，但必须整除
`logical_block_tokens`：

```text
logical_block_tokens % kernel_page_size == 0
```

此时 backend 会把一个 scheduler logical page 展开成多个 kernel page。某个 group
自己的一个 CacheBlock 覆盖多少原始 token，则由 `PagedCacheGroupSpec` 描述：

```text
group cache block tokens = rows_per_page × entry_stride_tokens
```

DeepSeek V4 的 compressed/state group 就会使用这种 group-specific 几何。因此，想
知道 scheduler 的统一 page 粒度时读取 `logical_block_tokens`；想知道某个 group 或
kernel 的真实行数和 token span 时，应读取对应 group spec 和 backend page size。

### 常见取值

| Recipe | `logical_block_tokens` |
| --- | --- |
| 普通 MHA/MLA/DSA/MSA | `config.page_size` |
| Qwen GDN | 128 |
| Inkling | 128 |
| Kimi K3 | 128 |
| DeepSeek V4 | 256 |

它不表示物理字节数、page 数量或最终准入容量：

- 物理父块字节数看 `lcm_block_bytes`；
- 某个 group 的 page 数看 `group.page_count`；
- scheduler 最终可接纳 token 数看 `CachePoolSpec.token_capacity`。

## `groups`：逻辑分页组

每个元素是一个 `CacheGroupLayout`：

```python
@dataclass(frozen=True)
class CacheGroupLayout:
    group_id: str
    cache_blocks_per_lcm_block: int
    page_count: int
```

一个 group 表示一组共享 page-id 空间和 packing 的缓存，例如：

- `full_attention`
- `sliding_attention`
- `linear_attention`
- `kvconv`
- `compressor_state`

各字段含义如下：

- `group_id`：group 的唯一名称；
- `cache_blocks_per_lcm_block`：一个 LCM 父块能够容纳多少个该 group 的子
  page，也叫 packing；
- `page_count`：该 group 的逻辑 page 总数，包含 null page 0。

Page 数量满足：

```text
page_count = 1 + num_lcm_blocks * cache_blocks_per_lcm_block
```

例如有 10 个 LCM 父块：

```text
full_attention  packing = 1 → page_count = 1 + 10 × 1 = 11
state           packing = 8 → page_count = 1 + 10 × 8 = 81
```

Full-attention page 较大，一个父块只能放一个；state page 较小，一个父块可以放
多个。Group id 是物理 plan 与 scheduler contract 的连接键，但 retention、sliding
window、history/state family 和 PD transfer policy 不存放在 `CacheGroupLayout` 中，
而是由 `CachePoolSpec.paged_cache_group_specs` 中同名的 `PagedCacheGroupSpec` 提供。

## `planes`：物理内存平面

每个元素是一个 `CachePlaneLayout`：

```python
@dataclass(frozen=True)
class CachePlaneLayout:
    plane_id: str
    bytes_per_lcm_block: int
    arena_offset_bytes: int
```

Plane 表示 arena 中一段独立、连续的物理区域：

- `plane_id`：物理平面的唯一名称；
- `bytes_per_lcm_block`：每个 LCM 父块在该 plane 中占用的字节数；
- `arena_offset_bytes`：整个 plane 在 arena 中的起始位置。

Arena 使用 plane-major 布局：

```text
arena
├── plane A：所有 LCM block 的 A 区域
├── plane B：所有 LCM block 的 B 区域
└── plane C：所有 LCM block 的 C 区域
```

不同 plane 顺序拼接、互不重叠。一个 plane 的总长度是：

```text
(num_lcm_blocks + 1) * bytes_per_lcm_block
```

### `arena_offset_bytes` 如何计算

`arena_offset_bytes` 是从 `CachePool.buffer[0]` 到该 plane 第一个字节的距离，单位是
bytes。它不是设备地址、Tensor id、page offset 或 field offset。

第一个 plane 从 0 开始；后续 plane 的起点等于前面所有 plane 完整长度的前缀和：

```text
first_plane.arena_offset_bytes = 0

next_plane_offset =
    current_plane_offset
    + (num_lcm_blocks + 1) * current_plane.bytes_per_lcm_block
```

`+1` 表示每个 plane 都包含一段 null parent 空间。因为 plane 总长度依赖实际的
`num_lcm_blocks`，容量无关的 `CacheLayout` 只保存 `plane_bytes`；调用
`with_num_lcm_blocks()` 生成 `CacheMemoryPlan` 时才计算 `arena_offset_bytes`。

例如有两个可用父块：

```text
num_lcm_blocks = 2

K plane:
  bytes_per_lcm_block = 1024
  arena_offset_bytes = 0
  total bytes = (2 + 1) × 1024 = 3072

V plane:
  bytes_per_lcm_block = 2048
  arena_offset_bytes = 3072
  total bytes = (2 + 1) × 2048 = 6144
```

对应的 plane-major arena 为：

```text
buffer[0:3072]       → K plane：null parent + 2 个可用 parent
buffer[3072:9216]    → V plane：null parent + 2 个可用 parent
```

因此定位一个具体 field 时需要逐层组合三种 offset：

```text
arena_offset_bytes   定位整个 plane
page_stride_bytes    定位 plane 内的逻辑 page
field_offset_bytes   定位 page 内的具体 field
```

完整寻址公式见后文 [Field 的实际字节地址](#field-的实际字节地址)。

Plane 还描述物理复用关系：不同 group 的 field 如果使用相同 `plane_id`，就会
用各自的 packing 和 page stride 解释同一段物理字节。例如一个 8 字节 plane
可以同时具有以下两种布局解释：

```text
group A：packing = 1，每页 8 字节
group B：packing = 4，每页 2 字节

┌──────────────────────────────┐
│        A page：8 bytes       │
├───────┬───────┬───────┬──────┤
│ B p0  │ B p1  │ B p2  │ B p3 │
│ 2B    │ 2B    │ 2B    │ 2B   │
└───────┴───────┴───────┴──────┘
```

因此同一 plane 的基础大小取各 group 所需空间的最大值，而不是把它们相加，
随后再满足对齐约束：

```text
plane_bytes = align_up(
    max(group_packing * group_payload_bytes_in_plane),
    plane_alignment,
)
```

哪些 field 可以使用同一 plane 由模型的 cache recipe 决定；scheduler 和运行时
契约则通过共享的 LCM 父块几何协调这些 overlay view。

## `fields`：具体 Tensor 字段

每个元素是一个 `CacheFieldLayout`：

```python
@dataclass(frozen=True)
class CacheFieldLayout:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    field_offset_bytes: int
    page_stride_bytes: int
```

Field 是具体的计算数据，例如：

```text
layer.0.k
layer.0.v
layer.3.latent_kv
layer.7.conv_state
layer.7.recurrent_state
```

每个 field 同时归属于一个 group 和一个 plane：

- `group_id` 决定 field 使用哪个 page id 空间和多少个 page；
- `plane_id` 决定 field 落在 arena 的哪段物理区域；
- `field_id` 是 `CachePool.field()` 使用的唯一查询名称；
- `shape` 是一个 child page 内的 Tensor 形状，不包含最外层 page 维度；
- `element_size` 是每个元素的字节宽度；plan 只记录整数几何，不依赖具体
  PyTorch dtype；
- `field_offset_bytes` 是 field 在一个 page 内的起始偏移；
- `page_stride_bytes` 是相邻两个逻辑 page 之间的字节距离。

一个 field 的单页有效数据大小是：

```text
payload_bytes = prod(shape) * element_size
```

例如：

```python
CacheFieldLayout(
    group_id="full_attention",
    field_id="layer.0.k",
    plane_id="unit.0.k",
    shape=(128, 8, 128),
    element_size=2,
    field_offset_bytes=0,
    page_stride_bytes=262144,
)
```

通过 `CachePool.field("layer.0.k", torch.bfloat16)` 绑定后，逻辑 Tensor shape
为：

```text
[group.page_count, 128, 8, 128]
```

最外层 page 维度来自 group，页内 shape 来自 field；它的 storage base 来自
plane，页与页之间按 `page_stride_bytes` 跳转。

## 三者如何关联

可以用一句话概括：

```text
group 决定“有多少页、如何调度”
plane 决定“这些页落在哪段物理内存、与谁复用”
field 决定“计算代码看到什么 Tensor”
```

完整映射关系如下：

```text
CacheGroupLayout
  │ 提供 page_count 和 packing
  ▼
CacheFieldLayout ──────→ field Tensor 的第 0 维和页内 shape
  │ 通过 plane_id 定位
  ▼
CachePlaneLayout
  │ arena_offset + page_stride + field_offset
  ▼
CachePool.buffer 中的实际字节
```

以普通 MHA 为例：

```text
field layer.0.k
  group = full_attention
  plane = unit.0.k

field layer.0.v
  group = full_attention
  plane = unit.0.v
```

K/V 使用相同的 group，因此共享同一个 page id 空间；K 和 V 位于不同 plane。
Scheduler 分配一个 `full_attention` page id 后，同一个 id 可以索引对应的 K field
和 V field。

## Plan 的整体容量

Arena 大小为：

```text
arena_bytes = (num_lcm_blocks + 1) * lcm_block_bytes
```

多出来的一个 block 是保留的 null block。类似地，每个 group 的 page 数为：

```text
page_count = 1 + num_lcm_blocks * cache_blocks_per_lcm_block
```

其中 page 0 是 null page。一个 LCM 父块可以为不同 group 容纳不同数量的子
CacheBlock，从而让尺寸不同的 KV、SWA 和 state page 共享统一的父块编号体系。

## 从 `CacheLayout` 创建 Plan

生产代码不会让 `CachePool` 临时拼装 plan。Recipe 先调用 `solve_cache_layout()` 求出
一个父块的容量无关布局，再根据显存预算计算 `num_lcm_blocks`，最后调用：

```python
plan = layout.with_num_lcm_blocks(num_lcm_blocks)
```

该方法完成三件事。

### 1. 展开 group 容量

对 `CacheLayout.group_packing` 中的每个 `(group_id, packing)` 创建：

```python
CacheGroupLayout(
    group_id=group_id,
    cache_blocks_per_lcm_block=packing,
    page_count=1 + num_lcm_blocks * packing,
)
```

`num_lcm_blocks` 必须是正整数，并且 `num_lcm_blocks * packing` 不能超过
32 位 kernel page id 的上限 `2^31 - 1`。

### 2. 展开 plane offset

Plane 使用 plane-major 布局。第一个 plane 从 0 开始，后续 plane 的 offset 按下面的
公式累加：

```text
next_plane_offset =
    current_plane_offset
    + (num_lcm_blocks + 1) * bytes_per_lcm_block
```

`+1` 表示每个 plane 都为 null parent 保留一段完整父块宽度。所有 plane 的完整区间
首尾相接，最终总长度等于 `arena_bytes`。

### 3. 复用 field 几何

`CacheLayout.fields` 已经包含每个 field 的 `field_offset_bytes` 和
`page_stride_bytes`。绑定容量不会改变一个 child page 内的布局，因此
`CacheMemoryPlan.fields` 直接复用这组 immutable field layout。

换句话说：

```text
CacheLayout 决定一个父块内部怎么切；
CacheMemoryPlan 决定把这种父块重复多少次以及各 plane 放在哪里。
```

## 查询 API

Plan 提供三个按 id 进行线性查找的方法：

```python
plan.group(group_id)   # -> CacheGroupLayout
plan.plane(plane_id)   # -> CachePlaneLayout
plan.field(field_id)   # -> CacheFieldLayout
```

找不到对应 id 时均抛出 `KeyError`。这些方法让调用方不需要依赖 tuple 的排列顺序：

- `CachePool.field()` 先用 `field()` 找字段，再用 `group()` 和 `plane()` 定位存储；
- runtime contract 用 `group()` 对齐 scheduler group 与物理 page count；
- 清零、PD 和 Host L2 transfer 根据 field/group/plane 查询生成字节区间。

Plan 不缓存字典索引。字段和 group 数量主要随模型层数增长，初始化与传输路径更看重
简单、不可变和易序列化的表示；高频 attention kernel 不会在每个 token 上调用这些
Python 查询方法。

## Field 的实际字节地址

`CacheFieldLayout` 给出了相对几何，`CachePool` 使用下面的公式计算某个 field、某个
逻辑 block 的 arena byte offset：

```text
byte_offset(field, block_id) =
    plane.arena_offset_bytes
    + plane.bytes_per_lcm_block
    - field.page_stride_bytes
    + block_id * field.page_stride_bytes
    + field.field_offset_bytes
```

其中：

- `plane.arena_offset_bytes` 定位物理 plane；
- `field.page_stride_bytes` 把逻辑 block id 转成 child-page 位移；
- `field.field_offset_bytes` 定位 page 内的具体字段；
- `plane.bytes_per_lcm_block - page_stride_bytes` 是 null page 的特殊偏移。

### 为什么 page 0 位于 null parent 的最后一个 child slot

假设某 plane 每个父块 8 字节，某 group 的 packing 为 4，因此 page stride 为 2
字节。它的布局是：

```text
null parent                     usable parent 0
┌────┬────┬────┬────┐          ┌────┬────┬────┬────┐
│空  │空  │空  │p0  │          │p1  │p2  │p3  │p4  │
└────┴────┴────┴────┘          └────┴────┴────┴────┘
 0    2    4    6               8   10   12   14
```

代入公式：

```text
block 0 offset = 8 - 2 + 0 × 2 = 6
block 1 offset = 8 - 2 + 1 × 2 = 8
block 4 offset = 8 - 2 + 4 × 2 = 14
```

因此每个 group 只暴露一个统一的 null page 0，而不是暴露 `packing` 个 null child
page；可用 page 1 则恰好从第一个可用父块开始。即便不同 group 的 packing 不同，它们
的 page 0 都落在各自解释下 null parent 的最后一个 child slot。

## `arena_bytes` 与不同容量概念

`arena_bytes` 是属性，不是存储字段：

```python
@property
def arena_bytes(self) -> int:
    return (self.num_lcm_blocks + 1) * self.lcm_block_bytes
```

几个容易混淆的容量如下：

| 名称 | 单位 | 是否包含 null | 含义 |
| --- | --- | --- | --- |
| `num_lcm_blocks` | parent blocks | 否 | 可用于真实请求的父块数量。 |
| `group.page_count` | child pages | 是，包含 page 0 | 某个 group 的逻辑 page-id 上界。 |
| `arena_bytes` | bytes | 是，包含 null parent | `CachePool` 实际分配的字节数。 |
| `CachePoolSpec.pool_size` | token slots | 否 | 最大 packing group 对应的物理 token 槽位数。 |
| `CachePoolSpec.token_capacity` | tokens | 否 | Scheduler 考虑并发和多 group 压力后的安全准入容量。 |

特别需要注意：`arena_bytes` 不能用 `pool_size * dtype.itemsize` 推导。Arena 可以包含
多个 plane、不同 element size、padding 和 overlay view，唯一权威值就是 plan 的
`arena_bytes`。

## `capacity_report()`：按 group 解释容量

多 group cache 无法只用一个 token 数准确描述容量：

- full-history group 按历史 token 持续增长；
- sliding-window group 的单请求驻留量受 window 限制；
- KDA/Mamba state group 更接近“每请求固定占几个 block”。

`capacity_report()` 因此按每个 group 自己的消费单位返回诊断信息：

```python
report = plan.capacity_report(
    window_tokens={"swa": 4096},
    per_request_blocks={"state": 2},
    max_num_seqs=32,
)
```

返回值形式为：

```python
{
    "group_id": {
        "unit": "tokens" | "requests",
        "capacity": int,
        "supported_requests": int | None,
        "dead_bytes": int | None,
        "binding_utilization": float,
    }
}
```

### 普通 full-history group

可用 page 数和 token 容量为：

```text
usable_pages = num_lcm_blocks * packing
token_capacity = usable_pages * logical_block_tokens
```

由于每个请求的上下文长度未知，`supported_requests` 为 `None`；该诊断不把未使用的
历史容量视为静态 dead bytes。

### Sliding-window group

传入 `window_tokens[group_id]` 后：

```text
supported_requests = token_capacity // window_tokens
```

如果还提供 `max_num_seqs`，超过 `max_num_seqs * window_tokens` 的静态 slab 容量无法
被活跃 window 使用，报告会按该 group 的 field payload 估算 `dead_bytes`。

### Per-request state group

传入 `per_request_blocks[group_id]` 后，容量单位变为 requests：

```text
supported_requests = usable_pages // blocks_per_request
```

提供 `max_num_seqs` 时，超过请求上限的 state page 同样会计入 `dead_bytes`。

### `binding_utilization`

每个 group 的 binding utilization 为：

```text
packing * sum(group field payload bytes) / lcm_block_bytes
```

它表示“把这个 group 绑定到一个共享父块时，实际 payload 占整个父块的比例”。共享
plane 按最宽 tenant 定尺寸，较窄 tenant 会留下 binding hole，因此该值可以用于发现
LCM overlay 带来的静态浪费。Registry 也会把它写入 cache geometry 诊断信息。

`capacity_report()` 是规划与诊断工具，不改变 plan，也不决定 scheduler 的最终
`token_capacity`；后者由具体模型 recipe 结合请求并发、调度 batch 和 state pressure
单独计算。

## 下游如何消费 Plan

同一份 `CacheMemoryPlan` 被多个运行时组件复用：

```text
CachePool
├── arena_bytes                         分配 uint8 buffer
├── field/group/plane                   建立 Tensor view
├── group.page_count                    发布 runtime contract
├── field/plane offset                  按 page 清零
├── fields + runtime dtype              构建 PD contract
└── planes/fields/groups                构建 Host L2 transfer layout
```

这种共享避免不同路径分别推导 offset。只要 recipe 生成的 plan 一致，计算、调度、清零
和传输看到的就是同一套物理几何。

## 关键不变量与常见误区

### Plan 是纯整数几何

Plan 只记录 `element_size`，不保存 `torch.dtype`。实际 dtype 由具体 pool 在调用
`field(field_id, dtype)` 时提供，且 `dtype.itemsize` 必须等于 plan 的
`element_size`。

### `num_lcm_blocks` 不包含 null parent

总 arena 使用 `num_lcm_blocks + 1`，但可用 group page 数只由
`num_lcm_blocks * packing` 贡献，额外再加唯一的 page 0。

### Group 不保存 retention

`CacheGroupLayout` 只保存 group id、packing 和 page count。Retention、sliding
window、history/state family 等 scheduler 语义位于 `PagedCacheGroupSpec`，两者在
`CachePool` 发布 runtime contract 时按 group id 对齐。

### Plane overlay 不是字段相加

共享同一 plane 的不同 group 是对同一字节区域的不同解释。Plane 宽度由最宽 binding
和对齐约束决定，不是把所有 group payload 简单相加。

### `page_stride_bytes` 可以大于 payload

允许 runtime stride 的 field 可以因为 plane 对齐或其他 tenant 更宽而带 padding；
只有 `CacheFieldSpec.exact_page_stride=True` 的 field 才要求两者严格相等。

### 应从 `CacheLayout` 生成生产 Plan

`CacheMemoryPlan` dataclass 自身没有 `__post_init__()` 去重新验证任意手工参数。生产
代码应当使用经过 `solve_cache_layout()` 校验的 `CacheLayout`，再调用
`with_num_lcm_blocks()`，不要手工拼接可能互相矛盾的 groups、planes 和 fields。
