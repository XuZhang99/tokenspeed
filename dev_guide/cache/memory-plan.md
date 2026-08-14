# CacheMemoryPlan：容量绑定的物理计划

[上一篇：CacheLayout](layout.md) · [返回目录](../cache.md) · [下一篇：Arena 与 field()](field-binding.md)

`CachePool` 不自行计算物理布局，而是执行 recipe 生成的 `CacheMemoryPlan`。
该类型定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)。

Plan 中最重要的三个集合分别处在不同层次：

```text
groups：调度和分页语义
planes：物理内存及复用关系
fields：计算代码看到的 Tensor 视图
```

除此之外，plan 还保存 `num_lcm_blocks`、`lcm_block_bytes` 和
`logical_block_tokens` 等整体几何信息。

## `groups`：逻辑分页组

每个元素是一个 `CacheGroupLayout`：

```python
@dataclass(frozen=True)
class CacheGroupLayout:
    group_id: str
    cache_blocks_per_lcm_block: int
    page_count: int
```

一个 group 表示一组具有相同分页、packing 和 retention 语义的缓存，例如：

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
多个。Group 是 scheduler 能看到的维度，scheduler 根据它管理 page id、retention、
sliding window 和 state snapshot 等语义。

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
