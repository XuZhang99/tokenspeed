# CacheLayout：容量无关的单父块布局

[上一篇：CachePool 总览](overview.md) · [返回目录](../cache.md) · [下一篇：CacheMemoryPlan](memory-plan.md)

`CacheLayout` 定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)，
表示一个 LCM 父块内部的完整字节几何，但不包含实际要分配多少个父块：

```python
@dataclass(frozen=True)
class CacheLayout:
    logical_block_tokens: int
    lcm_block_bytes: int
    group_packing: tuple[tuple[str, int], ...]
    plane_bytes: tuple[tuple[str, int], ...]
    fields: tuple[CacheFieldLayout, ...]
```

它处在 recipe 描述与最终物理 plan 之间：

```text
模型 cache recipe
    │ 生成 CacheFieldSpec
    ▼
solve_cache_layout(...)
    │ 求解单个 LCM 父块的 packing、plane 和 field 几何
    ▼
CacheLayout                         ← 容量无关
    │ 根据显存预算确定 num_lcm_blocks
    ▼
layout.with_num_lcm_blocks(N)
    ▼
CacheMemoryPlan                     ← 容量已绑定
    ▼
CachePool
```

## 输入：`CacheFieldSpec`

Recipe 先为模型的每个缓存组件创建 `CacheFieldSpec`：

```python
@dataclass(frozen=True)
class CacheFieldSpec:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    exact_page_stride: bool = True
    page_stride_alignment_bytes: int = 1
```

其中前三个 id 分别说明该字段属于哪个调度 group、叫什么名字、放在哪个物理
plane；`shape` 和 `element_size` 描述单个 child page 的 payload。

另外两个字段表达 kernel 对页 stride 的要求：

- `exact_page_stride=True`：kernel 隐式按 payload 大小跨页，求解后的
  `page_stride_bytes` 必须严格等于 `payload_bytes`；
- `exact_page_stride=False`：kernel 使用 Tensor 的运行时 stride，可以接受 page
  之间的 padding；
- `page_stride_alignment_bytes`：允许 padding 时仍需满足的字节对齐约束。

## `CacheLayout` 的字段

- `logical_block_tokens`：一个逻辑 cache block 表示多少个 token；
- `lcm_block_bytes`：一个完整 LCM 父块占多少字节，包含所有 plane，并满足所有
  group packing 的公倍数对齐要求；
- `group_packing`：`(group_id, packing)` 列表，表示一个父块包含多少个该 group
  的 child page；
- `plane_bytes`：`(plane_id, bytes_per_lcm_block)` 列表，表示每个 plane 在单个
  父块中的宽度；
- `fields`：已经求解完成的 `CacheFieldLayout`，包含每个 field 的页内 offset 和
  page stride。

此时还没有 `num_lcm_blocks`，所以也没有：

- 每个 group 的最终 `page_count`；
- 每个 plane 在完整 arena 中的 `arena_offset_bytes`；
- 最终 `arena_bytes`。

## `solve_cache_layout()` 做什么

`solve_cache_layout()` 只执行整数几何计算，不分配 Tensor。它主要完成：

1. 校验 field id、shape、element size 和 stride 约束；
2. 根据各 group 的 payload 比例推导 packing，或采用 recipe 显式指定的 packing；
3. 将同一 group、同一 plane 中的 field 排列起来，计算
   `field_offset_bytes`；
4. 对使用同一 plane 的不同 group 做 overlay，计算对齐后的 `plane_bytes`；
5. 计算 `page_stride_bytes = plane_bytes / group_packing`；
6. 将所有 plane 的大小求和并按 packing 的公倍数对齐，得到
   `lcm_block_bytes`；
7. 检查 exact-stride、最大 padding 比例和整数范围等不变量。

函数参数中的：

- `cache_blocks_per_lcm_block` 可以显式固定部分或全部 group packing；
- `alignment` 控制 plane/parent 的基础对齐；
- `max_padding_fraction` 限制为了统一布局而浪费的空间比例。

## 从 `CacheLayout` 到 `CacheMemoryPlan`

Recipe 根据显存预算和 `layout.lcm_block_bytes` 算出能够容纳的父块数，然后调用：

```python
memory_plan = layout.with_num_lcm_blocks(num_lcm_blocks)
```

该方法为每个 group 生成：

```text
page_count = 1 + num_lcm_blocks * packing
```

并为每个 plane 计算完整 arena 中的起始偏移：

```text
next_plane_offset =
    current_plane_offset
    + (num_lcm_blocks + 1) * bytes_per_lcm_block
```

其中 `+1` 为 null parent 预留空间。`with_num_lcm_blocks()` 还会检查父块数必须为
正整数，并保证 packing 展开后的 kernel page id 不溢出。

## 数值示例

假设 recipe 定义两个 field：

```python
CacheFieldSpec("history", "history.k", "plane.a", (4,), 1)
CacheFieldSpec("state", "state.ssm", "plane.b", (8,), 1)
```

并显式指定：

```text
history packing = 2
state   packing = 1
```

该示例调用 `solve_cache_layout()` 时使用 `logical_block_tokens=4` 和
`max_padding_fraction=1.0`，允许为了统一父块几何产生最多 100% padding。

求解单父块布局后得到：

```text
CacheLayout
├── logical_block_tokens = 4
├── group_packing
│   ├── history = 2
│   └── state   = 1
├── plane_bytes
│   ├── plane.a = 8 bytes
│   └── plane.b = 8 bytes
├── field page strides
│   ├── history.k = 8 / 2 = 4 bytes
│   └── state.ssm = 8 / 1 = 8 bytes
└── lcm_block_bytes = 16 bytes
```

如果随后调用 `layout.with_num_lcm_blocks(2)`，最终 plan 为：

```text
history page_count = 1 + 2 * 2 = 5
state   page_count = 1 + 2 * 1 = 3

plane.a arena_offset = 0
plane.b arena_offset = (2 + 1) * 8 = 24

arena_bytes = (2 + 1) * 16 = 48
```

这种两阶段设计把“模型字段应当如何排布”与“当前显存预算能分配多少容量”分开。
同一个 `CacheLayout` 可以用不同的 `num_lcm_blocks` 实例化成不同容量的
`CacheMemoryPlan`。
