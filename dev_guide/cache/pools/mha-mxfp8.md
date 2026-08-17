# MHATokenToKVPoolMXFP8

[子类索引](README.md) · [父类：MHA](mha.md)

源码：[`kv_cache/mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/mha.py)

该类在 MHA K/V 上增加 block-scaled MXFP8 data/scale view。普通 MHA recipe 当前
要求 `prefix_granularity == MXFP8_KV_SCALE_TILE_TOKENS == 128`。

## 字段

```text
layer.<id>.k / v          float8_e4m3fn
layer.<id>.k_scale        float8_e8m0fnu
layer.<id>.v_scale        float8_e8m0fnu
```

scale 的公共几何由 `recipes/plan.py::mxfp8_kv_scale_fields()` 唯一定义；每 32 个
head-dim value 共享 scale，因此 `head_dim % 32 == 0`。scale 保持 page-major，data
则是 token-row view。

`layer_plane_bindings` 在父类 K/V 之外增加 `k_scale_buffer` / `v_scale_buffer`。
`_layer_scale_view()` 会结合每层 served head 数和
`arena.kv_page_size` 生成 kernel 所需的 interleaved shape。

## 写入

`set_kv_buffer()` 要求已经量化的 `float8_e4m3fn` K/V 与非空 UE8M0 scale：

- data 以 `uint8` bit view scatter；
- scale 通过 `store_sf_interleaved()` 写入，或在非 interleaved 情况按 loc 写入；
- 异构 head 使用 `_layer_page_tokens()` 保持 scale 与 data 的 row 解释一致。

`quantize_and_set_kv_buffer()` 在 interleaved layout 且 `head_dim == 128` 时融合量化、
data store 与 scale scatter；返回 `False` 表示调用方应走分离 fallback，未发生写入。

## 不变量

- 不要把 BF16 直接交给 `set_kv_buffer()`；
- scale field shape 必须来自公共 helper，recipe 不应复制；
- field dtype 是 plan 的真实 FP8 dtype，`store_dtype` 只影响 writer；
- 异构 head 的 data/scale 必须用同一 layer page-token 几何。
