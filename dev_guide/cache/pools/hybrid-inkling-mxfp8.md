# HybridInklingTokenToKVPoolMXFP8

[子类索引](README.md) · [Inkling](hybrid-inkling.md) · [Hybrid MHA MXFP8](hybrid-mha-mxfp8.md)

源码：[`kv_cache/hybrid_inkling.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py)

这个空子类用 MRO 组合 Inkling checkpoint accessor 与 hybrid MXFP8 K/V：

```python
class HybridInklingTokenToKVPoolMXFP8(
    HybridInklingTokenToKVPool,
    HybridMHATokenToKVPoolMXFP8,
):
    pass
```

完整字段集由 recipe 一次声明：

```text
k, v, k_scale, v_scale
kvconv_k, kvconv_v, attnconv, mlpconv
```

MXFP8 scale shape 由 `mxfp8_kv_scale_fields()` 唯一定义，要求 P 是 128 的正倍数且
`head_dim % 32 == 0`；实际 Inkling 配置还需满足其 backend 约束。ShortConv dtype
规则与非 MXFP8 Inkling 相同。

K/V data/scale 走 MXFP8 writer，checkpoint 走 Inkling accessor，异构 head 走 MHA
row view。所有 view 已由同一个 `CacheArena` 构造，MRO 不参与物理分配。

overlay 与 recurrent checkpoint 使该类必须清零复用 block。PD/L2 schema 必须覆盖
data、scale 与 checkpoint 全部 field。
