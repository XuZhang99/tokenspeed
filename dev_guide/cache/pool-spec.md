# CachePoolSpec：Cache recipe 与运行时 Pool 的交接对象

[返回目录](../cache.md) · [CachePool 总览](overview.md) · [CacheMemoryPlan](memory-plan.md) · [运行时集成](runtime.md)

`CachePoolSpec` 定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/setup.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/setup.py)，
是 cache recipe 生成、`create_cache_pool()` 消费的不可变数据对象。它把已经绑定容量的
物理计划、逐层 cache 元数据、scheduler group 语义和模型特有选项放在一起，但本身
不分配显存，也不是运行时 cache pool。

```text
模型配置 + AttentionConfig + cache 显存预算
                    │
                    ▼
             prepare_cache_setup()
                    │ 选择模型 recipe
                    ▼
     CacheFieldSpec → CacheLayout → CacheMemoryPlan
                    │
                    ├── 逐层 group/type 信息
                    ├── scheduler group specs
                    ├── state dtype / KV head 信息
                    └── token admission capacity
                    ▼
               CachePoolSpec
                    │
                    ▼
             create_cache_pool()
                    │
                    ▼
       MHA / MLA / Hybrid CachePool 子类
```

可以把它概括为：

```text
CacheMemoryPlan 描述“字节如何排布”；
CachePoolSpec 描述“用哪个 Pool、各层怎样使用这些字节、scheduler 如何管理它们”。
```

## 类型定义

当前定义如下：

```python
@dataclass(frozen=True)
class CachePoolSpec:
    family: CacheModelFamily
    memory_plan: CacheMemoryPlan
    layer_types: tuple[str, ...]
    layer_group_ids: tuple[str, ...]
    paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...]
    state_field_dtypes: Mapping[str, torch.dtype]
    token_capacity: int
    layer_kv_head_counts: tuple[int, ...] | None = None
    pool_options: object | None = None
```

`frozen=True` 表示对象创建后不能原地修改。Recipe 生成的布局、调度语义和逐层映射
因而能够作为同一份事实来源传给 registry、factory、pool 和 scheduler。需要派生
target/draft 视图时，代码通过 `dataclasses.replace()` 创建新对象，而不是修改原对象。

## 字段详解

### `family`

`family` 描述 cache 的模型布局族，factory 用它和 `AttentionConfig` 的具体类型一起
选择 `CachePool` 子类。当前取值为：

```text
mha
mla
dsa
msa
qwen_gdn
inkling
kimi_k3
deepseek_v4
```

它不是 attention backend 名称，也不是 scheduler 的 history/state family。例如：

- `family="mha"` 配合 `MHAConfig` 创建普通 `MHATokenToKVPool`；
- `family="qwen_gdn"` 配合 `MHAConfig` 创建 hybrid MHA/GDN pool；
- `family="kimi_k3"` 配合 `MLAConfig` 创建 `HybridKDATokenToKVPool`；
- `family="deepseek_v4"` 直接选择 DeepSeek V4 专用 pool。

量化变体通常还需要读取 config。例如普通 MHA 的 `kv_cache_mxfp8` 决定创建
`MHATokenToKVPool` 还是 `MHATokenToKVPoolMXFP8`，不能只根据 `family` 判断。

### `memory_plan`

`memory_plan` 是已经绑定实际容量的 `CacheMemoryPlan`，包含：

- `num_lcm_blocks` 和 `lcm_block_bytes`；
- 每个 group 的 packing 与 `page_count`；
- 每个 plane 的 arena offset；
- 每个 field 的 shape、offset 和 page stride；
- 最终 `arena_bytes`。

它是物理几何的权威来源。`create_cache_pool()` 将它传给具体 pool，`CachePool` 再按
plan 分配一个 `uint8` arena，并通过 `field()` 建立 typed/strided Tensor view。

`CachePoolSpec` 不复制 plan 中的信息，也不会根据其他字段重新推导物理布局。

### `layer_types`

`layer_types` 保存逐层的逻辑 cache 类型，主要用于描述某一层是 full attention、
sliding attention、linear/state，或者模型 recipe 定义的其他类别。

典型值如下：

```text
("full_attention", "linear_attention", "full_attention", ...)
```

普通模型无法提供可靠的逐层 label 时，这个 tuple 可以为空；此时 group spec builder
把各层按 full-history 语义处理。非空时通常应与 `layer_group_ids` 等长。

该字段的内容由具体 recipe 解释，并不保证所有 family 使用同一套字符串词表。例如
DeepSeek V4 会把每层 compression ratio 转成字符串保存在这里。

### `layer_group_ids`

`layer_group_ids` 是每层对应的物理 cache group id：

```text
layer 0 → full_attention
layer 1 → linear_attention
layer 2 → full_attention
```

它解决的是“第 N 层应该使用 plan 中哪个 page-id 空间”。具体 pool 会保存这组映射，
hybrid backend 或 `LayerMappedKVPool` 再据此把 layer id 路由到正确的 group。

当 target 和 draft 被合并规划时，该 tuple 覆盖两边的全部层：target 层在前，draft
continuation 层在后。因此：

```text
len(layer_group_ids) = target cache layers + draft cache layers
```

它与 `memory_plan.groups` 的粒度不同：`layer_group_ids` 是逐层数组，plan 的
`groups` 是去重后的物理 group 集合。

### `paged_cache_group_specs`

该字段保存 recipe 为 scheduler 生成的 `PagedCacheGroupSpec`。每个 spec 描述一个
group 的逻辑行为，包括：

- `group_id`：与 plan group 对应的唯一名称；
- `retention`：`full_history` 或 `sliding_window`；
- `rows_per_page` 和 `entry_stride_tokens`；
- `sliding_window_tokens`；
- `family`：scheduler 意义上的 `history` 或 `state`；
- `transfer_policy`：PD 场景下传完整 suffix 或最新 snapshot；
- `cache_blocks_per_lcm_block`：每个 LCM 父块内的 child page 数。

这里保存的是 group 语义，不保存最终 page count。Pool 发布 runtime contract 时，会：

1. 按 `group_id` 在 `memory_plan.groups` 中找到物理 group；
2. 用 plan 中的 packing 覆盖 spec 的 `cache_blocks_per_lcm_block`；
3. 从 plan 读取每个 group 的 `page_count`；
4. 生成 `PagedCacheRuntimeContract`。

因此两者的职责分别是：

```text
paged_cache_group_specs：为什么保留、保留多少历史、怎样传输
memory_plan.groups：实际有多少页、一个父块 packing 多少子页
```

每个发布的 group spec 都必须在 memory plan 中存在，而且 `group_id` 不能重复。

### `state_field_dtypes`

`state_field_dtypes` 是 `field_id → torch.dtype` 的映射，用来补充纯整数物理 plan
无法表达的运行时 dtype：

```python
{
    "layer.1.conv": torch.bfloat16,
    "layer.1.ssm": torch.float32,
}
```

`CacheMemoryPlan` 只记录 `element_size`，同样的字节宽度可能对应不同 dtype；hybrid
MHA 和 KDA pool 绑定 conv、SSM、recurrent state field 时，需要这个映射选择正确的
Tensor dtype。没有独立 state field 的普通 MHA/MLA/DSA/MSA 和 DeepSeek V4 recipe
通常传空映射。

### `token_capacity`

`token_capacity` 是 runtime contract 向 scheduler 公布的可接纳 token 容量。它由
recipe 在考虑以下因素后计算：

- cache 显存预算；
- LCM 父块大小与各 group packing；
- 最大请求数、上下文长度和调度 batch；
- decode/overlap 期间需要保护的 page；
- state snapshot、sliding window 等模型特有压力；
- 用户配置的 token 上限。

它与下面的 `pool_size` 不是同一个概念。`pool_size` 是最大 packing 对应的几何 token
槽位数，`token_capacity` 是 scheduler 可以安全接纳的 token 数。普通单 group cache
中二者可能相同；Kimi K3、DeepSeek V4 等多 group cache 中，后者可能因为并发和
状态页需求而更小。

### `layer_kv_head_counts`

这是可选的逐层 KV head 数，当前主要服务 Inkling 这类不同层具有不同 KV head 数的
模型。Tuple 中保存的是模型侧、TP 切分前的逐层数量；MHA pool 会结合 allocation
宽度和 `attn_tp_size`，计算当前 rank 实际使用的 head 数。

物理 field 可以按最大 head 数分配，而较窄层通过 reshape 把相同字节解释成更多的
token row。该字段为空表示所有层使用统一 head 数。

如果非空，它必须与 `layer_group_ids` 等长；target/draft 合并时，两边必须同时提供
逐层 head 数或同时不提供。

### `pool_options`

`pool_options` 是模型专用扩展点，默认是 `None`。当前 DeepSeek V4 使用
`DeepseekV4PoolOptions` 携带包含逐层 compression ratio 等信息的 V4 layout：

```python
pool_options=DeepseekV4PoolOptions(layout=merged_layout)
```

Factory 在创建 DeepSeek V4 pool 前会检查它的具体类型。通用字段应直接加入
`CachePoolSpec`；只有某个 family 独有、其他 pool 不应理解的数据才适合放在这里。

## `pool_size` 属性

`pool_size` 是根据 memory plan 动态计算的属性：

```python
max_packing = max(
    group.cache_blocks_per_lcm_block
    for group in memory_plan.groups
)
pool_size = (
    memory_plan.num_lcm_blocks
    * max_packing
    * memory_plan.logical_block_tokens
)
```

它表示最大 packing group 对应的非 null token 槽位数，不是 arena 字节数，也不包含
null parent/page 0。实际显存字节数应读取 `memory_plan.arena_bytes`。

例如：

```text
num_lcm_blocks = 10
max_packing = 4
logical_block_tokens = 128

pool_size = 10 × 4 × 128 = 5120 token slots
```

Registry 使用它作为 `max_num_tokens` 和 pool 构造参数 `size`；scheduler 的安全准入
上限仍以 `token_capacity` 为准。

## 在哪里创建

运行时入口在 attention registry。它先完成 cache 显存 profiling，然后调用：

```python
cache_setup = prepare_cache_setup(
    family=cache_family,
    ...,
    cache_budget_bytes=cache_memory,
)
spec = cache_setup.spec
```

`prepare_cache_setup()` 根据 `family` 分派到不同 recipe：

| Family | Spec 创建路径 |
| --- | --- |
| `mha`、`mla`、`dsa`、`msa` | `prepare_ordinary_cache()` → `_ordinary_setup()` |
| `qwen_gdn` | `prepare_qwen35_cache()` → `build_hybrid_cache_setup()` |
| `inkling` | `prepare_inkling_cache()` → `build_hybrid_cache_setup()` |
| `kimi_k3` | `prepare_kimi_k3_cache()` |
| `deepseek_v4` | `prepare_deepseek_v4_cache()` |

虽然构造位置不同，步骤基本一致：

1. 为 target 以及可选 draft 生成 cache fields 和逐层 group；
2. 把 draft 当作 target 后面的 continuation layers 合并；
3. 调用 `solve_cache_layout()` 得到容量无关的 `CacheLayout`；
4. 根据显存预算和运行时压力确定 `num_lcm_blocks`；
5. 调用 `layout.with_num_lcm_blocks()` 得到 `CacheMemoryPlan`；
6. 构造 `CachePoolSpec`，再包装到 `CacheSetup` 返回。

`CacheSetup` 还保存 `num_draft_layers`、原始 cache budget 和固定 workspace 大小。这些
属于初始化过程，不需要传给具体 pool，所以没有放进 `CachePoolSpec`。

## `layer_view()`：派生 target/draft 计算视图

默认情况下，一个 `CachePoolSpec` 覆盖合并后的 target 与 draft 全部层。异构 draft
需要使用不同的 `family` 和 pool 计算接口时，registry 调用 `layer_view()` 派生两个
视图：

```python
target_spec = spec.layer_view(
    first_layer=0,
    num_layers=num_target_layers,
)

draft_spec = spec.layer_view(
    first_layer=num_target_layers,
    num_layers=num_draft_layers,
    family=draft_family,
    publish_runtime_contract=False,
)
```

`layer_view()` 会切片：

- `layer_types`；
- `layer_group_ids`；
- `layer_kv_head_counts`（如果存在）。

它可以覆盖 `family`，并通过 `publish_runtime_contract=False` 清空 draft view 的
`paged_cache_group_specs`。它不会切分或重建 `memory_plan`，也不会改变
`token_capacity`、state dtype 和 pool options。

```text
merged CachePoolSpec
├── target view：target 层元数据 + 完整 runtime group specs
└── draft view：draft 层元数据 + 空 group specs
                  │
                  └── 共享 target 已创建的 backing arena 和 runtime contract
```

这样能够保证：

- target 和 draft 使用同一份物理 plan；
- 只由 target pool 发布一次 scheduler contract；
- draft pool 只建立自己的计算 field view，不重复分配 arena；
- draft 本地 layer 0 可以通过 `field_layer_offset` 映射到合并 plan 中的全局层。

当前 backing view 只支持普通 MHA 或 MLA pool；hybrid family 不会因为存在
`layer_view()` 就自动获得共享 view 支持。

## Factory 如何消费 Spec

`create_cache_pool(spec, config, ...)` 不创建或修改 spec，而是读取它：

```text
spec.family                  → 选择 Pool family
spec.memory_plan             → 传入 CachePool，决定 arena 几何
spec.pool_size               → Pool 的逻辑 size
spec.layer_types             → 逐层计算类型
spec.layer_group_ids         → 逐层 group 路由
spec.paged_cache_group_specs → 发布 scheduler runtime contract
spec.state_field_dtypes      → 绑定 state Tensor dtype
spec.token_capacity          → runtime contract 的准入容量
spec.layer_kv_head_counts    → 异构 KV head view
spec.pool_options            → 模型专用构造信息
```

Factory 还会校验 config 类型与 family 的组合。例如 `family="mha"` 需要
`MHAConfig`，`family="kimi_k3"` 需要 `MLAConfig`，DeepSeek V4 则必须携带正确类型的
`pool_options`。

## 关键不变量与常见误区

### Spec 不等于 Pool

`CachePoolSpec` 是纯描述对象；`CachePool` 才持有 arena、Tensor view 和运行时状态。
创建 spec 不会占用 KV Cache 显存。

### Spec 不等于 Memory Plan

`CacheMemoryPlan` 只描述整数化的物理几何。`CachePoolSpec` 在它之外补充 pool family、
逐层映射、runtime dtype 和 scheduler 语义。

### `layer_group_ids` 不等于 `paged_cache_group_specs`

前者每层一个元素，允许多个层重复引用同一 group；后者每个唯一 scheduler group
一个元素。

### `pool_size` 不等于 `arena_bytes`

`pool_size` 的单位是 token slots，`arena_bytes` 的单位是字节。底层分配严格采用
`memory_plan.arena_bytes`。

### `pool_size` 不一定等于 `token_capacity`

前者来自最大 packing 的物理几何，后者是考虑多 group 并发压力后的 scheduler
准入上限。需要决定可调度容量时应使用 `token_capacity`。

### Runtime contract 必须只发布一次

主 pool 使用完整的 `paged_cache_group_specs` 发布 contract；共享 backing arena 的
draft view 必须传空 tuple 并继承主 pool 的 contract，否则构造时会报错。
