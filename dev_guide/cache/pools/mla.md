# MLATokenToKVPool

[子类索引](README.md) · [Arena 与 `field()`](../field-binding.md)

源码：[`kv_cache/mla.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/mla.py)

## 定位与创建条件

`MLATokenToKVPool` 直接继承 `CachePool`，实现 Multi-head Latent Attention 的
压缩 KV cache：

```text
CachePool
└── MLATokenToKVPool
```

Factory 在 `config` 是精确的普通 `MLAConfig` 且 `spec.family == "mla"` 时创建。
普通 MLA 支持 `field_layer_offset` 和 `backing_pool`，因此可以作为异构 draft view
绑定到 target 的合并 arena。

## 核心几何

MLA 不保存完整的多头 K/V，而保存：

```text
kv_cache_dim = kv_lora_rank + qk_rope_head_dim
```

普通模式下每层只有一个 field：

```text
layer.<global_layer_id>.latent_kv
```

原始 field shape 为：

```text
[group.page_count, page_size, 1, kv_cache_dim]
```

绑定后展平为：

```text
kv_buffer[layer].shape =
[group.page_count * page_size, 1, kv_cache_dim]
```

和 MHA 一样，第 0 个 page 是 null page。

## `per_token_head` 量化布局

当 `quant_method == "per_token_head"` 时，一个 layer 被拆成三个 field：

```text
layer.<id>.latent_kv       FP8 latent/nope
layer.<id>.latent_scale    FP32 scale
layer.<id>.rope_k          model dtype rope component
```

对应运行时 tuple：

```text
(
  latent_kv:    [slots, 1, kv_lora_rank],
  latent_scale: [slots, 1, 1],
  rope_k:       [slots, 1, qk_rope_head_dim],
)
```

Scale 由 latent 部分的绝对值最大值计算，latent 被量化到 FP8；rope 部分使用同一
scale 缩放后以 `model_dtype` 保存。

## Getter 的兼容语义

- `get_key_buffer(layer_id)`：普通模式返回完整 latent+rope buffer；量化模式返回
  三元组；
- `get_value_buffer(layer_id)`：普通模式返回 latent KV-LoRA 前缀，量化模式返回
  `(latent_kv, latent_scale)`；
- `get_kv_buffer(layer_id)`：组合上述两个结果。

这套 K/V 命名是为了兼容通用 attention pool ABI。MLA 物理上没有普通 MHA 那样
彼此独立的 K buffer 和 V buffer。

公开 getter 同样会等待 `layerwise_load_tracker`。如果层在 hybrid KDA pool 中是
state layer，`kv_buffer[layer_id]` 为 `None`，普通 MLA getter 会明确报错。

## 写入接口

`set_kv_buffer()` 用于兼容通用 PagedAttention writer：

- 普通模式只把 `cache_k` 写入 latent buffer，`cache_v` 不单独保存；
- `per_token_head` 模式把 `cache_k` 拆成 latent/nope 与 rope，计算 scale 后分别
  写入三个 field。

MLA backend 更常使用：

```python
set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope, sanitize=False)
```

普通模式调用 Triton kernel 把 nope/rope 写入一个联合 buffer；bitwise store dtype
场景会先转成逻辑 cache dtype，再以原始 word view 写入。`sanitize=True` 会在写入
路径处理 NaN，避免 dummy slot 污染 decode。

## 读取并重建 MLA 分量

```python
get_mla_kv_buffer(layer, loc, dst_dtype=None)
```

普通模式通过 Triton kernel 从联合 buffer 分离 nope 与 rope。量化模式读取 FP8
latent、FP32 scale 和 rope field，并执行：

```text
cache_k_nope = latent_fp8 * scale
cache_k_rope = rope_stored * scale
```

结果转换到调用方指定的 `dst_dtype`。

## 传输布局

传统连续传输元数据随量化方式变化：

- 普通模式：每层一个 latent buffer；
- `per_token_head`：每层三个物理 buffer，按 component-major 顺序排列；
- `get_layerwise_buf_info_offsets()` 将每层映射到对应的 1 个或 3 个传输条目。

统一 PD cache contract 仍由 `CachePool` 基类根据 plan 生成。

## 关键不变量

- `layer_group_ids` 必须与 `layer_num` 等长；
- `kv_cache_dim` 必须等于 latent 与 rope 宽度之和；
- plan 必须包含与 `quant_method` 对应的一组 field；
- backing draft view 必须共享完全相同的 memory plan；
- 不要把 `get_value_buffer()` 理解成独立 V cache，它只是 latent 前缀的兼容 view。
