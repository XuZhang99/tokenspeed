# LCM 两级 Cache 分配机制

本文说明当前 cache planner 如何把不同 group 的 CacheBlock 打包进统一 parent。对象
级细节见 [Cache 开发文档](cache.md)，逻辑/物理命名规范见
[`docs/design/cache-concepts.md`](../docs/design/cache-concepts.md)。

## 1. 为什么需要 LCM parent

混合模型的 cache 单位并不等宽：

- MHA/MLA history 是逐 token row；
- sliding window 只保留有限历史；
- recurrent/conv state 每隔若干 token 保存 snapshot；
- Kimi K3 的 MLA 与 KDA state、DeepSeek V4 的压缩链有不同 payload；
- MXFP8 还包含独立 scale plane。

若为每组预留独立大池，某些 group 会因为 per-request state 或 window 上限留下大量
静态空洞。LCM 布局将若干 group 的 child CacheBlock 打包到一个字节统一的 parent：

```text
LCM parent
├─ plane 0：group A × K_A child blocks
├─ plane 1：group B × K_B child blocks
└─ ...
```

同一 parent 在 allocator 中一次只归一个 group 所有；不同 group 的 Tensor view 可以
overlay 同一字节几何，但不会把两组 live data 同时写到同一 parent。

## 2. 三类粒度

必须区分：

| 名称 | 单位 | 所有者 |
| --- | --- | --- |
| `prefix_granularity`（P） | token | prefix identity / memory plan |
| `block_granularity` | token | 每个 `CacheGroupSpec` |
| `cache_blocks_per_lcm_block` | child blocks / parent | `CacheLayout` / `CacheMemoryPlan` |

row group 的 `block_granularity = rows_per_page × entry_stride_tokens`；snapshot state
group 的 `block_granularity = checkpoint_granularity`。kernel 的 `kernel_page_size` 是
第四个独立概念，来自 backend registry，不属于 scheduler 或 LCM packing。

## 3. 唯一流水线

[`recipes/base.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/base.py)
中的 `CacheRecipe.setup()` 声明唯一顺序：

```text
layers ──group──▶ declarations ──pack──▶ CacheLayout ──bind──▶ CacheMemoryPlan
```

### 3.1 layers

recipe 提供 target 后接 draft continuation 的：

- `layer_types`：attention/state label；
- `group_ids`：每层落在哪个物理 group。

普通 family 用默认 `groups()`；Inkling 和 DeepSeek V4 因为有非 per-layer group，覆盖
该 seam。

### 3.2 group

[`recipes/spec.py::group()`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py)
一次遍历同时生成每个 group 的逻辑 spec 与 field 列表：

```text
(CacheGroupSpec, CacheFieldSpec[])[]
```

group id 只写在 spec 中，field 通过外层 pair 归属；同一批 pair 同时交给 planner 和
runtime contract，消除了两份 group 枚举不一致的可能。

### 3.3 pack

[`recipes/plan.py::pack()`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)
求一个 parent 的容量无关 `CacheLayout`：

- 每个 group 的 packing；
- 每个 plane 的 parent 宽度；
- field 在 plane 内的 offset 与 page stride；
- 总 `lcm_block_bytes`。

`CacheFieldSpec.exact_page_stride=True` 的字段要求 stride 恰好等于 payload；柔性字段
可以使用 padding，但 kernel 必须读取 runtime stride。planner 同时校验 alignment 与
`max_padding_fraction`。

### 3.4 bind

family 根据显存预算与并发求出 parent 数，然后：

```python
plan = layout.bind(num_lcm_blocks)
```

生成：

```text
group.page_count = 1 + N × packing
arena_bytes       = (N + 1) × lcm_block_bytes
```

额外 parent/block 是 null placeholder。

## 4. `CacheRecipe` 扩展 seam

新增或修改 family 时，优先覆盖以下统一 seam：

| seam | 作用 |
| --- | --- |
| `layer_types`, `group_ids` | layer vocabulary |
| `fields_for_layer()` | 一个 layer 的 field bytes |
| `groups()` | 只有非 per-layer group 才覆盖 |
| `prefix_granularity`, `alignment`, `max_padding_fraction` | planner 参数 |
| `packing()` | 钉死或调整 group packing |
| `check_layout()` | pack 后、bind 前的 family invariant |
| `num_lcm_blocks()` | 从预算求 parent 数 |
| `parents_needed()`, `token_capacity()` | 非简单容量模型 |
| `workspace_bytes()` | arena 之外的固定 workspace |
| `pool_options()` | family-only runtime layout |

子类应使用 `@override`；不要覆盖 `setup()` 并复制流水线。

## 5. Family 差异

### Ordinary：MHA / MLA / DSA / MSA

[`recipes/ordinary.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/ordinary.py)
统一处理四类：

- 每组 packing 固定为 1；
- parent span 等于 P；
- parent 数从 profile 的 `cache_cell_size × storage_layers` 求得；
- MHA 可附加 MXFP8 scale；MLA 可拆 latent/scale/rope；DSA/MSA 附加 index-K。

hybrid slab 的实际 storage layer 数由 `hybrid_slab_group_size()` 统一决定，避免 profile
与 planner 使用不同 divisor。

### Kimi K3

[`recipes/kimi_k3.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py)
把 full-attention packing 固定为 12，并用 MLA plane 字节宽度推导三个 KDA state
group 的 packing。`check_layout()` 要求 KDA 完全骑在 MLA plane 内，不产生额外 plane。

K3 的 state working set 与 history demand 不符合简单乘积，因此覆盖
`parents_needed()`，再用基类 `_capacity_from_parents()` 的单调二分反推 admission
capacity。

### DeepSeek V4

[`recipes/deepseek_v4.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py)
整体声明 SWA、compressed KV、compressor state 和 indexer group。kernel byte geometry
来自
[`deepseek_v4_geometry.py`](../python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py)，
不由 recipe 反向定义。

V4 packing 采用 power-of-two 约束；capacity 同样使用 family-specific parent demand。
P 可以是 kernel page 的正倍数，不应写死为旧文档中的 256。

### Inkling

[`recipes/inkling.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py)
先用公共 layer walk 建 attention group，再追加 KVConv/HiddenConv 两个 checkpoint
column。checkpoint field 使用 flexible stride 与 K/V plane overlay，workspace 包含
ShortConv ring。

### Qwen GDN

[`recipes/qwen35.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py)
将重复的 linear layer position 拆成 state group，attention layer 声明 K/V，state
layer 声明 conv/ssm。ReplaySSM 与普通 state 的 workspace/shape 差异由 recipe 处理。

## 6. Arena、Pool 与 scheduler

`CacheLayout.bind()` 的 plan 先交给 `CacheArena`：

```text
CacheArena
├─ 分配 (N + 1) × parent_bytes
├─ eager materialize 所有 typed field view
└─ 发布 CacheRuntimeContract
```

target/draft `CachePool` 只绑定各自 layer window。scheduler bridge 从 contract 读取
group spec、page count、packing，并生成 C++ `CacheGroupConfig`。C++ scheduling 在 token
单位工作；LCM packing 只进入 cache allocator。

## 7. 清零、PD 与 L2

- hybrid/V4 pool 声明 `requires_page_zeroing`；scheduler 返回的新 block 由
  `CacheArena.zero_blocks()` 按 field byte range 清零；
- PD 直接传 `CacheMemoryPlan + CacheGroupSpec + CacheTransferSchema`，dtype 已在 plan；
- Host L2 的 `CacheTransferLayout` 同样从 plan 和 pool layer selection 建立；
- 任何路径都应复用 `field_page_byte_offset()`，不能复制 parent/child 定位公式。

## 8. 数值示意

若 P=128，两个 group packing 分别为 12 和 3，N=100：

```text
history page_count = 1 + 100 × 12 = 1201
state   page_count = 1 + 100 × 3  = 301
arena parent count = 101          # 包含 null parent
```

两个 group 的 block table slot span 仍由各自 `CacheGroupSpec.block_granularity` 决定；
不能用 `12 × P` 或 `3 × P` 替代逻辑 span。packing 只说明一个 parent 容纳多少物理
child block。

## 9. 维护检查表

- 新 family 注册在 `recipes/setup.py::_RECIPES`；
- group spec 与 fields 必须成对声明；
- dtype 写入 `CacheFieldSpec`，不得由 pool 再提供；
- kernel geometry 放在 attention/kernel 层，不应反向依赖 recipe；
- capacity 只从 `scheduler_limits` 读并发；
- target/draft 使用一个合并 plan、一个 arena、一个 contract；
- 修改 plan wire 字段时同步 PD 序列化和测试。
