# CachePool 设计概览

`CachePool` 定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/base.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/base.py)，
是 TokenSpeed KV Cache 的物理存储基类。它根据预先生成的
`CacheMemoryPlan` 分配一块扁平的字节 arena，再把 arena 中的不同区域暴露成
MHA、MLA 和 recurrent state 等计算路径需要的 Tensor view。

```text
cache recipe
    │ 生成布局和调度描述
    ▼
CachePoolSpec + CacheMemoryPlan
    │
    ▼
CachePool：一块共享的 uint8 arena
    ├── field("layer.0.k")   → K Tensor view
    ├── field("layer.0.v")   → V Tensor view
    ├── field("state.ssm")   → state Tensor view
    ├── runtime_contract     → scheduler
    ├── PD contract          → 跨节点传输
    └── transfer layout      → Host L2/offload
```

## 职责边界

`CachePool` 负责：

- 按 `CacheMemoryPlan` 懒分配物理 arena；
- 为 plan 中的 field 创建 typed、strided Tensor view；
- 发布 scheduler 所需的分页缓存运行时契约；
- 按 group 和 page 清零缓存；
- 描述 PD 分离和 Host L2/offload 的传输布局；
- 支持 target/draft 计算视图共享同一个物理 arena。

它不负责决定请求应当使用哪个 page，也不实现具体 attention kernel 的 KV
读写。page id 由 scheduler 管理；`get_key_buffer()`、`get_value_buffer()`、
`get_kv_buffer()` 和 `set_kv_buffer()` 等计算接口由具体子类实现。

因此可以概括为：`CachePool` 管理“字节放在哪里以及如何共享”，具体子类管理
“attention kernel 以什么 Tensor 形状读写这些字节”。

## 继承体系

当前 `kv_cache/` 目录中的生产类继承关系如下：

```text
CachePool
├── MHATokenToKVPool
│   ├── MHATokenToKVPoolMXFP8
│   ├── MSATokenToKVPool
│   └── HybridMHATokenToKVPool
│       ├── HybridMHATokenToKVPoolMXFP8
│       └── HybridInklingTokenToKVPool
│           └── HybridInklingTokenToKVPoolMXFP8
├── MLATokenToKVPool
│   ├── DSATokenToKVPool
│   └── HybridKDATokenToKVPool
└── HybridDeepseekV4TokenToKVPool
```

其中两个 MXFP8 hybrid 类使用多重继承组合“模型布局”和“量化存储”能力：

- `HybridMHATokenToKVPoolMXFP8` 同时继承 `HybridMHATokenToKVPool` 和
  `MHATokenToKVPoolMXFP8`；
- `HybridInklingTokenToKVPoolMXFP8` 同时继承
  `HybridInklingTokenToKVPool` 和 `HybridMHATokenToKVPoolMXFP8`。

同文件中的 `LayerMappedKVPool` 不在这棵继承树中：它是组合式 wrapper，通过
转发调用把模型层 id 映射到内部 pool 的 layer slot，本身不继承 `CachePool`。

### `MHATokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/mha.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/mha.py)，
是普通 Multi-Head Attention 的计算接口。

它为每层绑定独立的 K/V field：

```text
layer.<id>.k
layer.<id>.v
```

并把它们转换成 `[slot, kv_head, head_dim]` 形式。`set_kv_buffer()` 负责 dtype
转换并调用 `store_kv_cache` 写入指定 slot；`get_key_buffer()` 和
`get_value_buffer()` 则提供 attention backend 使用的 per-layer view。

该类还支持不同层具有不同 KV head 数。物理分配按最大 head 数规划，窄 head
层通过 `_layer_row_view()` 把相同字节重新解释成更多、更窄的 token row，避免
改变 plan 的 page stride。

### `MHATokenToKVPoolMXFP8`

这是 MHA pool 的 MXFP8 存储变体。除 K/V 数据 field 外，每层还绑定：

```text
layer.<id>.k_scale
layer.<id>.v_scale
```

K/V 数据使用 `float8_e4m3fn`，每 32 个 head-dim 元素配一个
`float8_e8m0fnu` scale。它支持两种写入方式：

- `set_kv_buffer()` 接收已经量化的数据和 scale；
- `quantize_and_set_kv_buffer()` 在布局满足要求时融合量化、K/V 写入和 scale
  scatter。

当 page size 为 128，或异构 head 布局可换算为 128 的整数倍时，scale 使用
block-scaled attention kernel 需要的 interleaved 格式；否则使用 flat scale
布局。

### `MLATokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/mla.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/mla.py)，
是 Multi-head Latent Attention 的计算接口。它不像 MHA 那样保存完整 K/V，
而是保存压缩后的 latent cache。

普通模式下，每层绑定：

```text
layer.<id>.latent_kv
```

并将其解释成 `[slot, 1, kv_lora_rank + qk_rope_head_dim]`。在
`per_token_head` 量化模式下，一个 layer 会拆成：

```text
layer.<id>.latent_kv
layer.<id>.latent_scale
layer.<id>.rope_k
```

`set_mla_kv_buffer()` 和 `get_mla_kv_buffer()` 负责在 latent/nope 与 rope
两部分之间写入、读取以及必要的量化/反量化。为兼容通用 attention 接口，
`get_key_buffer()` 返回完整 latent cache，`get_value_buffer()` 返回其中的
KV-LoRA 部分。

### `DSATokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py)，
继承 `MLATokenToKVPool`，在 MLA latent cache 之外为每层增加稀疏检索使用的：

```text
layer.<id>.index_k
```

`index_k` 按 token-group 量化成 FP8，并保存对应的 FP32 scale。一个 page 内采用
block-split 布局，即先连续存放所有 FP8 key，再连续存放 scale，而不是每个 token
交错存储。`set_index_k_buffer()` 负责量化和 scatter，`gather_index_k()` 为非分页
prefill scoring 路径重新收集连续的 key/scale Tensor。

该类也把 index-K 纳入显存统计、PD 连续 buffer 信息和 layer-wise transfer
offset。

### `MSATokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/msa.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/msa.py)，
继承 `MHATokenToKVPool`，用于 MiniMax Sparse Attention。它保留普通 MHA K/V，
同时只为 `indexed_layer_ids` 中的稀疏层绑定 key-only side cache：

```text
layer.<id>.index_k
```

`get_index_k_buffer()` 会拒绝访问未配置索引缓存的层。当前传统
`get_contiguous_buf_infos()` 传输 ABI 尚未扩展到该 side cache，因此会明确抛出
`NotImplementedError`；物理布局和通用 contract 仍由 `CachePool` 管理。

### `HybridMHATokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py)，
主要对应 `qwen_gdn` family。它让普通 MHA history 与 recurrent state 共享同一个
arena：

- attention layer 绑定 `layer.<id>.k` 和 `layer.<id>.v`；
- state layer 绑定 `layer.<id>.conv` 和 `layer.<id>.ssm`。

State layer 在继承的 `k_buffer`/`v_buffer` 中保留 `None`，调用方通过
`get_state_buffers()` 或 `get_component()` 访问 conv/recurrent state。该类还提供
layer 到 cache group 的映射，并严格检查 page stride 是否满足 attention/GDN
kernel 的连续性要求。

因为 history 与 state 会复用物理页，它设置
`paged_cache_requires_page_zeroing = True`，新 page 必须通过 `zero_new_pages()`
清零。它不使用旧的连续 buffer 传输接口，跨节点传输统一走
`get_pd_cache_contract()`。

### `HybridMHATokenToKVPoolMXFP8`

该类把 `HybridMHATokenToKVPool` 的 history/state 混合布局与
`MHATokenToKVPoolMXFP8` 的 block-scaled K/V 存储能力组合起来。除绑定 state
field 和 attention K/V 外，还绑定每层的 K/V scale field，并复用 MXFP8 的写入
接口。它额外要求 `head_dim` 能被 32 整除。

### `HybridInklingTokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py)，
继承 hybrid MHA pool，并增加 Inkling ShortConv 的 checkpoint view：

```text
layer.<id>.kvconv_k
layer.<id>.kvconv_v
layer.<id>.attnconv
layer.<id>.mlpconv
```

`kvconv_k/v` 使用 BF16。`attnconv` 和 `mlpconv` 的 dtype 由
`INKLING_FP8_SCONV` 控制，默认 BF16，启用后为 `float8_e5m2`。

`HybridInklingTokenToKVPoolMXFP8` 不增加新方法，而是通过多重继承在上述
ShortConv checkpoint view 基础上叠加 MXFP8 attention K/V 存储。

### `HybridKDATokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_kda.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_kda.py)，
主要对应 `kimi_k3` family。它让 MLA latent KV 与 KDA recurrent state 共享同一个
arena：

- MLA layer 绑定 `layer.<id>.latent_kv`；
- KDA state layer 绑定 `layer.<id>.conv_state` 和
  `layer.<id>.recurrent_state`。

它通过 `get_component()` 统一暴露 `latent_kv`、`conv_state` 和
`recurrent_state`，并提供 layer 到 group 的映射。KDA 当前不支持 MLA 的
`per_token_head` 量化模式；新 page 必须清零，PD 传输只能使用 cache contract，
不使用旧的连续 buffer/layer-wise offset ABI。

### `HybridDeepseekV4TokenToKVPool`

定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py)，
直接继承 `CachePool`，因为 DeepSeek V4 的缓存组成与普通 MHA/MLA 差异较大。

它根据每层 compression ratio 绑定以下 field：

- 所有层：`layer.<id>.swa`；
- ratio 大于 1：`compressed_kv` 和 `compressor_state`；
- ratio 等于 4：额外绑定 `indexer_kv` 和 `indexer_state`。

SWA、compressed KV、compressor state 和 CSA indexer 由不同 paged-cache group
管理；`indexer_kv` 与对应 compressed-KV group 共享 page table 和 page budget。

为兼容通用接口，`get_key_buffer()` 和 `get_value_buffer()` 都返回 SWA buffer，
但标准 `set_kv_buffer()` 不可用，实际写入由 V4 attention helper 完成。该类还负责：

- 不同压缩率对应的 block size；
- compressed/indexer/state buffer 的强类型访问；
- scheduler state-group page 使用情况诊断；
- 新 page 清零；
- 通过统一 cache contract 进行传输。

它同样设置 `paged_cache_requires_page_zeroing = True`。

### Factory 选择关系

具体 pool 由
[`python/tokenspeed/runtime/layers/attention/kv_cache/factory.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/factory.py)
根据 attention config、`CachePoolSpec.family` 和量化配置选择：

| Config / family | 创建的 pool |
| --- | --- |
| `MHAConfig` / `mha` | `MHATokenToKVPool` 或 MXFP8 变体 |
| `MLAConfig` / `mla` | `MLATokenToKVPool` |
| `DSAConfig` / `dsa` | `DSATokenToKVPool` |
| `MSAConfig` / `msa` | `MSATokenToKVPool` |
| `MHAConfig` / `qwen_gdn` | `HybridMHATokenToKVPool` 或 MXFP8 变体 |
| `MHAConfig` / `inkling` | `HybridInklingTokenToKVPool` 或 MXFP8 变体 |
| `MLAConfig` / `kimi_k3` | `HybridKDATokenToKVPool` |
| `deepseek_v4` | `HybridDeepseekV4TokenToKVPool` |

## CacheLayout：容量无关的单父块布局

`CacheLayout` 定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)，
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

### 输入：`CacheFieldSpec`

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

### `CacheLayout` 的字段

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

### `solve_cache_layout()` 做什么

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

### 从 `CacheLayout` 到 `CacheMemoryPlan`

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

### 数值示例

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

## Memory Plan

`CachePool` 不自行计算物理布局，而是执行 recipe 生成的 `CacheMemoryPlan`。
该类型定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)。

Plan 中最重要的三个集合分别处在不同层次：

```text
groups：调度和分页语义
planes：物理内存及复用关系
fields：计算代码看到的 Tensor 视图
```

除此之外，plan 还保存 `num_lcm_blocks`、`lcm_block_bytes` 和
`logical_block_tokens` 等整体几何信息。

### `groups`：逻辑分页组

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

### `planes`：物理内存平面

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

### `fields`：具体 Tensor 字段

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

### 三者如何关联

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

### Plan 的整体容量

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

## Arena 的分配

物理 buffer 在第一次绑定 field 时由 `_ensure_buffer()` 懒分配：

```python
self.buffer = torch.zeros(
    self.plan.arena_bytes,
    dtype=torch.uint8,
    device=self.device,
)
```

Buffer 始终是一维 `uint8` Tensor。实际显存大小由 `plan.arena_bytes` 决定，
并不是 `size * dtype.itemsize`。使用字节 buffer 可以在同一个 arena 中容纳不同
dtype 的 field，也方便以统一的原始字节布局进行清零和传输。

## `field()`：创建计算视图

`field(field_id, dtype)` 是 `CachePool` 的核心方法。它执行以下步骤：

1. 懒分配底层 arena；
2. 检查 `field_id` 是否存在于 plan；
3. 检查运行时 dtype 的字节宽度是否符合 plan；
4. 根据 field 的 byte offset 和 page stride，通过 `as_strided()` 创建 Tensor view；
5. 将 view 缓存在 `_fields`，后续调用返回同一个对象。

简化后的逻辑如下：

```python
view = buffer.view(dtype).as_strided(
    (group.page_count, *field.shape),
    (
        field.page_stride_bytes // field.element_size,
        *inner_contiguous_strides,
    ),
    field_byte_offset // field.element_size,
)
```

这个过程不会复制数据。所有 field view 的 storage 都来自同一个
`CachePool.buffer`，但可以拥有不同的 dtype、shape、offset 和 page stride。

例如 MHA pool 会把每一层的 K/V field 转换为 kernel 熟悉的形状：

```python
self.k_buffer = [
    self.field(f"layer.{layer_id}.k", self.store_dtype)
    .view(-1, self.head_num, self.head_dim)
    for layer_id in range(self.layer_num)
]
```

## 构造参数

几个容易混淆的参数如下：

- `size`：逻辑 pool size。基类主要将其作为元数据，以及未显式传入
  `token_capacity` 时的缺省值；它不是 arena 的字节数。
- `dtype`：KV 的逻辑 dtype。对于部分 FP8 dtype，底层 `store_dtype` 会改为
  `uint8`，以规避 Torch 写入算子的限制。
- `device`：arena 所在设备。
- `page_size`：一个逻辑 cache block 对应的 token 数，通常等于
  `plan.logical_block_tokens`。
- `rank`：当前并行 rank 的元数据。
- `memory_plan`：物理布局和显存分配的权威来源。
- `paged_cache_group_specs`：各 group 的 family、retention、rows-per-page 等调度语义。
- `token_capacity`：向 scheduler 公布的有效 token 容量。
- `backing_pool`：让当前对象作为另一个 pool 的共享计算视图。
- `field_layer_offset`：把当前 view 的局部 layer id 映射到合并 plan 中的全局 layer。
- `pd_disaggregation_enabled`：是否允许发布 PD 分离传输契约。

## Scheduler 运行时契约

`_publish_runtime_contract()` 将 recipe 给出的逻辑 group spec 与 plan 中的真实
packing/page count 对齐，生成 `PagedCacheRuntimeContract`：

```python
PagedCacheRuntimeContract(
    block_size=page_size,
    num_lcm_blocks=plan.num_lcm_blocks,
    token_capacity=token_capacity,
    group_specs=aligned_group_specs,
    group_page_counts=group_page_counts,
)
```

Scheduler 通过该契约获得：

- 缓存组集合；
- 每组的 page 数；
- 每个 LCM 父块包含多少个该组的子 page；
- group 的 history/state、retention 和 sliding-window 等语义；
- 可调度的有效 token 容量。

`CachePool` 因此是物理内存布局与调度器逻辑分页之间的契约边界。

## 共享 backing pool

Target 和异构 draft 可以共享同一个 arena。带 `backing_pool` 构造的 view：

- 不分配新 buffer；
- 复用 backing pool 的 `buffer` 和 `_fields` 注册表；
- 继承 backing pool 的 `runtime_contract`；
- 通过 `field_layer_offset` 绑定 draft 对应的 continuation layer。

构造顺序必须是 target-first：backing pool 必须已经绑定 field 并完成 arena
分配。共享 view 也不能重复发布 `paged_cache_group_specs`，只能继承 backing pool
已经发布的运行时契约。

## 清零与传输

`CachePool` 提供两种清零方式：

- `zero_blocks()`：只清零指定 group 的指定 CacheBlock，按 plan 计算原始字节区间；
- `clear_kv_buffers()`：清空整个 arena，主要用于 sleep/wake 后修复重新映射的存储。

纯 attention pool 默认不要求 page 重用时清零。KV 与 recurrent state 共享物理
页面的 hybrid pool 会设置：

```python
paged_cache_requires_page_zeroing = True
```

这样可以避免旧 state 的尾部数据污染新请求。

传输相关接口包括：

- `pd_contract()`：生成 prefill/decode 分离所需的跨节点原始 slab 契约；
- `get_pd_cache_contract()`：在启用 PD disaggregation 后返回上述契约；
- `cache_transfer_layout()`：生成 Host L2/offload 所需的字节布局。

`pd_contract()` 要求 plan 中的全部 field 已经绑定，因为构建传输契约时需要知道
每个 field 的实际运行时 dtype。

## 创建与使用流程

完整生命周期可以概括为：

1. Cache recipe 根据模型结构和显存预算生成 `CachePoolSpec` 与
   `CacheMemoryPlan`；
2. `create_cache_pool()` 根据 family/config 选择具体 pool 子类；
3. `CachePool.__init__()` 保存 plan，并发布 scheduler runtime contract；
4. 子类在初始化时调用 `field()`，首次调用触发 arena 分配并建立所有计算 view；
5. Scheduler 根据 runtime contract 分配 page id；
6. Attention backend 通过具体子类的 `set_kv_buffer()` 写入，通过
   `get_*_buffer()` 取得对应 view；
7. PD、Host L2 和 sleep/wake 路径复用同一个 plan 描述进行传输或清零。
