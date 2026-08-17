# MLATokenToKVPool

[子类索引](README.md) · [Arena 与字段绑定](../field-binding.md)

源码：[`kv_cache/mla.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/mla.py)

`MLATokenToKVPool` 是 MLA 的计算视图。factory 在 `MLAConfig + family=mla` 时创建；
KDA 与 DSA 分别扩展 state 和 sparse index。

## 字段布局

普通模式每层一个：

```text
layer.<id>.latent_kv [P, 1, kv_lora_rank + qk_rope_head_dim]
```

arena 将其折成 `[slots, 1, latent_width]`。

`quant_method == "per_token_head"` 时拆成：

```text
latent_kv    [P, 1, kv_lora_rank]       cache dtype
latent_scale [P, 1, 1]                  float32
rope_k       [P, 1, qk_rope_head_dim]   model dtype
```

`layer_plane_bindings` 同时列出三类 plane；未规划的项为 `None`。绑定后普通模式的
`kv_buffer[layer]` 是 Tensor，量化模式是三元组。

## 接口

- `set_mla_kv_buffer()` 把 nope/rope 写入联合 latent，或拆分量化写入；
- `get_mla_kv_buffer()` 取回并在量化模式下反量化；
- 通用 `set_kv_buffer()` / `get_key_buffer()` / `get_value_buffer()` 是兼容 ABI，
  物理上并不存在 MHA 式独立 K/V；
- getter 可等待 layer-wise load tracker。

target/draft 共享由外部传入同一个 `CacheArena` 实现，draft 的局部 layer 通过
`field_layer_offset` 定位 continuation field；没有 `backing_pool` 参数。

## 不变量

- `layer_group_ids` 必须覆盖全部局部 layer；
- plan 的 field 集必须与 quant method 对应；
- hybrid state layer 的 latent list 项可以为 `None`，应改走 state accessor；
- `get_value_buffer()` 只是 latent 前缀兼容 view，不是独立 V cache。
