# CacheMemoryPlan：容量绑定的物理几何

[上一篇：CacheLayout](layout.md) · [返回目录](../cache.md) · [下一篇：Arena 与字段绑定](field-binding.md)

[`recipes/plan.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)
定义纯 Python、torch-free、可序列化的 cache 字节计划。它既驱动 GPU arena，也直接
进入 PD wire contract。

## 数据结构

```python
@dataclass(frozen=True)
class CacheMemoryPlan:
    prefix_granularity: int
    lcm_block_bytes: int
    num_lcm_blocks: int
    groups: tuple[CacheGroupLayout, ...]
    planes: tuple[CachePlaneLayout, ...]
    fields: tuple[CacheFieldLayout, ...]

@dataclass(frozen=True)
class CacheGroupLayout:
    group_id: str
    cache_blocks_per_lcm_block: int
    page_count: int

@dataclass(frozen=True)
class CachePlaneLayout:
    plane_id: str
    bytes_per_lcm_block: int
    arena_offset_bytes: int

@dataclass(frozen=True)
class CacheFieldLayout:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    dtype: str
    field_offset_bytes: int
    page_stride_bytes: int
```

`element_size` 和 `payload_bytes` 都由 `CacheFieldLayout.dtype` 与 `shape` 计算，不再
存一份可与 dtype 冲突的整数。

## `prefix_granularity` 不是通用 page size

`prefix_granularity`（简称 `P`）是 scheduler 做 prefix hashing/matching 的 identity
grain，单位为逻辑 token。它也是当前 arena 的 KV page convention 来源，因此
`CacheArena.kv_page_size` 从 plan 读取同一个值；但每个 group 的调度 span 仍由
`CacheGroupSpec.block_granularity` 决定，可以小于或不同于 `P`。

必须区分：

| 数量 | 所属域 | 含义 |
| --- | --- | --- |
| `prefix_granularity` | scheduler logical | prefix 身份边界 |
| `CacheGroupSpec.block_granularity` | group logical | 一个 block-table slot 跨多少 token |
| `kernel_page_size` | attention kernel | 一个 kernel page 跨多少 token |
| `cache_blocks_per_lcm_block` | physical | 一个 parent 装多少该组 CacheBlock |

`kernel_page_size` 来自 `kernel_page_sizes.py` 或显式 config，不能从 P 猜测。Python
映射层负责把 group block table 转成 kernel page table。

## parent、group 与 null block

`num_lcm_blocks` 只统计可用 parent；arena 额外保留一个 null parent：

```text
arena_bytes = (num_lcm_blocks + 1) × lcm_block_bytes
```

一个 group 的 block 数：

```text
page_count = 1 + num_lcm_blocks × cache_blocks_per_lcm_block
```

这里字段仍沿用 `page_count`，表示物理 CacheBlock id 空间，包括保留的 block 0。
block 0 不计入 admission capacity，并映射到 null parent 的末尾 child slot。

packing 最大的 group 能看到最多 block；arena 的便捷 token-slot 几何为：

```text
num_lcm_blocks × max_packing × prefix_granularity
```

这只是最细 group 的物理 row 容量。scheduler 的实际 admission 使用
`CachePoolSpec.token_capacity`。

## plane 与 overlay

plane 是一个 parent 内可被多个 group 以不同 stride 解释的字节列。
`bytes_per_lcm_block` 是该列在一个 parent 中的宽度，`arena_offset_bytes` 是整列在完整
arena 中的起点：

```text
plane 0: (N + 1) × plane0_parent_bytes
plane 1: (N + 1) × plane1_parent_bytes
...
```

同一 plane 上不同 group 的 `page_stride_bytes` 通常为：

```text
plane.bytes_per_lcm_block / group.packing
```

因此 group 可以 overlay 同一 parent 的物理字节，但 allocator 保证同一时刻 parent
只按一个 group 的 block 所有权使用。overlay 是静态布局复用，不是让两组 live 数据
同时占用同一字节。

## field

`CacheFieldLayout` 同时描述：

- 所属物理 group；
- 所在 plane；
- 单个 block 的 shape 与 dtype；
- 在 group 的 plane payload 内的 field offset；
- 相邻 block view 的字节 stride。

单个 field payload：

```text
product(shape) × dtype.itemsize
```

`exact_page_stride` 只存在于输入 `CacheFieldSpec`。经过 `pack()` 后，它已经转化成
`page_stride_bytes == payload_bytes` 这一不变量，无需在 plan 中保留布尔值。

## 字节定位

`field_page_byte_offset(field_id, page_id)` 是所有清零与传输路径应复用的定位函数：

```text
plane_offset
+ plane_parent_bytes
- field_page_stride
+ page_id × field_page_stride
+ field_offset
```

它先校验 `page_id` 位于对应 group 的 `[0, page_count)`，再返回 arena owner buffer
中的绝对字节 offset。不要在 pool、PD 或 L2 中复制这套算式。

## `CacheLayout.bind()` 是唯一容量绑定点

`pack()` 只描述一个 parent；family 结合显存预算与 scheduler limit 得到 parent 数，
然后调用：

```python
memory_plan = layout.bind(num_lcm_blocks)
```

`bind()` 填入 group page count 和 plane arena offset。生产代码不应手工拼接
`CacheMemoryPlan`；单测或 wire decode 直接构造时，也必须保持上述不变量。

## dtype 传播

常规 recipe 使用 `cache_dtype_name()` 把 torch dtype 转成稳定字符串。通过普通
elementwise scatter 写入且 torch 不支持 FP8 writer 时，使用
`scatter_stored_dtype_name()` 将 field 声明为 `uint8`；由 dtype-aware kernel 写入的
MXFP8 data/scale 则保留真实 FP8 dtype。

这保证：

```text
recipe field dtype
  = plan field dtype
  = arena Tensor dtype
  = PD contract field dtype
```

pool 的 `store_dtype` 不参与物理计划，只控制输入写入时的 bit reinterpretation。

## `capacity_report()`

`CacheMemoryPlan.capacity_report()` 用 group 自己的消费单位报告容量：

- full-history：token capacity；
- sliding-window：token capacity、按 window 估算 supported requests 与 dead bytes；
- state：按每请求 block 数估算 request capacity。

它还报告 `binding_utilization`，用于观察一个 group 在共享 parent 中实际使用的字节
比例。多组 cache 不应被压成单一「token 容量」来解释；最终 admission 仍以 recipe
求出的 `token_capacity` 为准。

## 维护检查表

- field dtype 必须在 recipe 中声明，不能在 arena/pool 再传一次；
- group packing 只从 layout/plan 读取；
- null parent 与 block 0 必须始终保留；
- 新增字节定位逻辑应调用 `field_page_byte_offset()`；
- group 逻辑 span、prefix grain、kernel page 与 physical packing 不得混名；
- PD wire 兼容性变化必须同时考虑 plan dataclass 的序列化/反序列化。
