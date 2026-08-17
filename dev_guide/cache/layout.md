# CacheLayout：`group → pack → bind`

[上一篇：总览](overview.md) · [返回目录](../cache.md) · [下一篇：CacheMemoryPlan](memory-plan.md)

布局阶段位于
[`recipes/spec.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py)
与
[`recipes/plan.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)。
它只做整数几何，不分配 Tensor。

## 输入一：`CacheGroupSpec`

一个 group 同时声明 retention、family、PD policy 和一种逻辑形状：

```python
@dataclass(frozen=True)
class CacheGroupSpec:
    group_id: str
    retention: Literal["full_history", "sliding_window"]
    rows_per_page: int | None = None
    entry_stride_tokens: int | None = None
    sliding_window_tokens: int | None = None
    family: Literal["history", "state"] = "history"
    transfer_policy: Literal["full_suffix", "latest_snapshot"] | None = None
    checkpoint_granularity: int | None = None
```

两种形状互斥：

- row geometry：`rows_per_page × entry_stride_tokens`，用于真正的 paged KV；
- checkpoint geometry：`checkpoint_granularity`，用于 recurrent/conv state snapshot。

所有调用方都可读取 `block_granularity`；只有 row geometry 才能读取 `page_size`。
snapshot state 没有 page，错误读取会抛 `TypeError`。

## 输入二：`CacheFieldSpec`

```python
@dataclass(frozen=True)
class CacheFieldSpec:
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    dtype: str
    exact_page_stride: bool = True
    page_stride_alignment_bytes: int = 1
```

与旧结构相比有两个关键变化：

- field 不带 `group_id`；它属于外层 `(CacheGroupSpec, fields)` 对中的 group，避免
  group id 被写两遍；
- field 直接带可序列化的 torch dtype 名，`element_size` 从 dtype 推导，避免布局和
  运行时 dtype 分叉。

`exact_page_stride=True` 表示 kernel 隐式按 payload 大小跨 block；为 `False` 时
kernel 必须读取 Tensor 的运行时 stride，planner 可利用共享 plane 的 padding。

## `group()`

`recipes/spec.py::group()` 对层只遍历一次，并按 `group_ids` 的首次出现顺序返回：

```text
((CacheGroupSpec, (CacheFieldSpec, ...)), ...)
```

它同时完成：

- layer label → retention/family；
- sliding window 校验；
- 相同 group 的字段聚合和 occurrence 编号；
- row/checkpoint geometry 声明；
- PD 模式下的 transfer policy。

Inkling 和 DeepSeek V4 的 group 不是纯 per-layer 结构，因此覆盖 recipe 的
`groups()` seam；它们仍返回同一种 pair，继续进入公共 `pack()`。

## `pack()`

`pack(groups, prefix_granularity=..., ...)` 生成容量无关的：

```python
@dataclass(frozen=True)
class CacheLayout:
    prefix_granularity: int
    lcm_block_bytes: int
    group_packing: tuple[tuple[str, int], ...]
    plane_bytes: tuple[tuple[str, int], ...]
    fields: tuple[CacheFieldLayout, ...]
```

`pack()` 的主要步骤：

1. 校验 group 唯一、field 唯一、shape/dtype/对齐合法；
2. 将每个字段与其外层 group id 配对并确定 field 在 group plane 内的 offset；
3. 对未钉死的 group 先按总 payload 比例估算 packing，再用共享 plane 上
   exact-stride 字段的字节比例求约束；
4. 柔性 stride group 尽量填入已有 plane 的 slack；
5. 计算每个 plane 的 parent 宽度和每个 field 的 page stride；
6. 校验 exact stride、padding fraction 和对齐；
7. 得到一个 LCM parent 的 `lcm_block_bytes`。

`cache_blocks_per_lcm_block` 可由 family recipe 全部或部分钉死。packing 是物理布局
的答案，不属于 `CacheGroupSpec`。

## `bind()`

容量确定后只调用：

```python
plan = layout.bind(num_lcm_blocks)
```

对每个 group：

```text
page_count = 1 + num_lcm_blocks × packing
```

对每个 plane，`arena_offset_bytes` 依次累加：

```text
(num_lcm_blocks + 1) × bytes_per_lcm_block
```

额外的 `+1` 是 null parent。`bind()` 还检查 block id 不超过 kernel 的 int32 上限。

## 最小示例

```python
groups = (
    (
        CacheGroupSpec(
            group_id="history",
            retention="full_history",
            rows_per_page=4,
            entry_stride_tokens=1,
        ),
        (CacheFieldSpec("layer.0.k", "k", (4, 8), "uint8"),),
    ),
    (
        CacheGroupSpec(
            group_id="state",
            retention="full_history",
            family="state",
            checkpoint_granularity=4,
        ),
        (
            CacheFieldSpec(
                "layer.1.ssm", "state", (16,), "float32",
                exact_page_stride=False,
            ),
        ),
    ),
)
layout = pack(groups, prefix_granularity=4, alignment=16)
plan = layout.bind(32)
```

生产代码应通过 `CacheRecipe.setup()` 进入这条路径；不要手工构造一份 group spec，
再用另一套字段列表或 packing 与它做事后 reconciliation。
