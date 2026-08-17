# CachePoolSpec：Recipe 的交付对象

[返回目录](../cache.md) · [总览](overview.md) · [CacheMemoryPlan](memory-plan.md) · [运行时集成](runtime.md)

[`recipes/setup.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/setup.py)
中的 `CachePoolSpec` 是 recipe 向 `CacheArena` 和 pool factory 交付的不可变描述。它
不分配显存，也不是一个 runtime pool。

## 当前定义

```python
@dataclass(frozen=True)
class CachePoolSpec:
    family: CacheModelFamily
    memory_plan: CacheMemoryPlan
    layer_types: tuple[str, ...]
    layer_group_ids: tuple[str, ...]
    cache_group_specs: tuple[CacheGroupSpec, ...]
    token_capacity: int
    layer_kv_head_counts: tuple[int, ...] | None = None
    pool_options: object | None = None
```

旧的 `paged_cache_group_specs` 已改名为更一般的 `cache_group_specs`：snapshot state
没有 page。旧的 `state_field_dtypes` 已删除：每个 field 的 dtype 已经写入
`CacheMemoryPlan`。

## 字段语义

### `family`

factory 的模型布局族：

```text
mha, mla, dsa, msa, qwen_gdn, inkling, kimi_k3, deepseek_v4
```

它不是 attention backend 名，也不是 `CacheGroupSpec.family` 的 `history/state`。
factory 还会结合具体 `BaseAttnConfig` 类型和量化开关选择 pool 类。

### `memory_plan`

容量已绑定的唯一物理几何，包含：

- `prefix_granularity`、parent 大小与 parent 数；
- 每组 page count 和 packing；
- plane 在完整 arena 中的 offset；
- 每个 field 的 group、shape、dtype、offset 和 stride；
- `arena_bytes`。

`create_cache_arena()` 直接用它分配 owner buffer；pool 必须通过 `arena.plan` 读取，
不得在自身保存副本。

### `layer_types` 与 `layer_group_ids`

二者按 global layer 排列：target 在前，draft continuation 在后。

- `layer_types` 是 family 的逻辑 label，例如 `full_attention`、
  `sliding_attention`、`linear_attention`；某些 ordinary family 可为空；
- `layer_group_ids` 是每层实际分配表对应的 group id，必须覆盖全部 cache layer。

一组 layer 可以共享一个 group；反过来，一个 model 也可以把重复的 linear layer
位置拆成多个 state group。plan 的 `groups` 是去重后的物理 group 集合，而
`layer_group_ids` 是逐层映射，两者不是同一个粒度。

### `cache_group_specs`

由 `CacheRecipe.groups()` 与字段一起声明并直接发布。每个 `CacheGroupSpec` 记录：

- group id、retention、history/state family；
- row geometry 或 checkpoint geometry；
- sliding window；
- PD transfer policy。

它不记录 packing 或 page count；这两项由 `memory_plan.groups` 单独拥有。arena 用同一
批 declaration 的 spec 和 plan 组合 runtime contract，因此无需事后交叉校验两套
独立构建的 group 集合。

### `token_capacity`

scheduler 可安全 admission 的 token 上限。默认 recipe 使用：

```text
num_lcm_blocks × max(group packing) × prefix_granularity
```

Kimi K3、DeepSeek V4 等多组 demand 不符合简单乘积时，会覆盖
`parents_needed()` / `token_capacity()`，通过公共的单调二分反推容量。所有 family
只从 `CacheRecipe.scheduler_limits` 读取 concurrency，避免 sizing 两套口径。

`token_capacity` 不等于某个具体 group 的 block 数，也不等于 arena 字节数。

### `layer_kv_head_counts`

可选的逐层、TP 前 KV head 数。Inkling 等异构 head 模型用它把按最大 head width
分配的字节重新解释成窄 head、更多 token row 的局部 view。若存在，长度必须覆盖
全部 target+draft layer。

### `pool_options`

只用于单一 family 的 pool 构造参数。当前 DeepSeek V4 使用
`DeepseekV4PoolOptions(layout=...)`，其中 layout 来自
`attention/deepseek_v4_geometry.py`，并支持 `layer_view()` 切片。通用信息应加入
显式字段，不能滥用 `pool_options`。

## `layer_view()`

```python
view = spec.layer_view(
    first_layer=target_layers,
    num_layers=draft_layers,
    family=draft_family,
)
```

它只切分 per-layer 计算元数据：

- `layer_types`；
- `layer_group_ids`；
- `layer_kv_head_counts`；
- 支持切片的 model-specific options。

它保留完整的 `memory_plan`、`cache_group_specs` 和 `token_capacity`，因为这些属于共享
arena，而不是局部 pool view。arena 已经从原始合并 spec 发布唯一 contract；派生
view 不会再发布。

## `CacheSetup`

```python
@dataclass(frozen=True)
class CacheSetup:
    spec: CachePoolSpec
    num_draft_layers: int
    cache_budget_bytes: int
    fixed_workspace_bytes: int
```

`num_target_layers` 从总 layer 数减去 draft layer 数得到。draft 被视为合并模型的
continuation layer，这一规则统一了同构和异构 draft 的 field 命名与物理所有权。

## 从 recipe 到 runtime

```text
CacheRecipe.setup()
  ├─ groups = self.groups()
  ├─ layout = pack(groups, ...)
  ├─ num_lcm_blocks = self.num_lcm_blocks(layout)
  └─ CachePoolSpec(
       memory_plan=layout.bind(num_lcm_blocks),
       cache_group_specs=tuple(spec for spec, _ in groups),
       ...
     )

CachePoolSpec
  ├─ create_cache_arena() → allocation + contract
  └─ create_cache_pool()  → target/draft compute view
```

维护时不要从 `memory_plan.fields` 重新生成 group spec，也不要从
`cache_group_specs` 重新推导 packing；两者分别拥有逻辑语义和物理几何。
