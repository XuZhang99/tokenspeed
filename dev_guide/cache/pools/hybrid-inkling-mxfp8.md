# HybridInklingTokenToKVPoolMXFP8

[子类索引](README.md) · [Inkling](hybrid-inkling.md) · [Hybrid MHA MXFP8](hybrid-mha-mxfp8.md)

源码：[`kv_cache/hybrid_inkling.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py)

## 多重继承定位

该类本身没有新增方法，而是通过 MRO 组合两套能力：

```python
class HybridInklingTokenToKVPoolMXFP8(
    HybridInklingTokenToKVPool,
    HybridMHATokenToKVPoolMXFP8,
):
    pass
```

```text
HybridInklingTokenToKVPool
  → ShortConv checkpoint accessor、HiddenConv dtype

HybridMHATokenToKVPoolMXFP8
  → hybrid arena、MXFP8 K/V data + scale、page zeroing
```

Factory 在 `MHAConfig + family == "inkling" + kv_cache_mxfp8=True` 时选择它。

## 完整字段集合

每层包含：

```text
MXFP8 attention:
  layer.<id>.k
  layer.<id>.v
  layer.<id>.k_scale
  layer.<id>.v_scale

ShortConv checkpoints:
  layer.<id>.kvconv_k
  layer.<id>.kvconv_v
  layer.<id>.attnconv
  layer.<id>.mlpconv
```

Data K/V 使用 `float8_e4m3fn`，scale 使用 `float8_e8m0fnu`；KVConv checkpoint
仍是 BF16，HiddenConv checkpoint 由 `INKLING_FP8_SCONV` 决定。

## Recipe 几何限制

Inkling MXFP8 recipe 明确要求：

```text
logical_block_tokens % 128 == 0
head_dim == 128
scale block size == 32
```

Scale field 页内 shape 为：

```text
[
  layer_kv_heads,
  logical_block_tokens / 128,
  32,
  4,
  4,
]
```

这与 `_layer_scale_view()` 和 FA4 interleaved scale ABI 对齐。

## 初始化 MRO

`HybridInklingTokenToKVPool.__init__()` 调用 `super()` 后，在 MRO 中进入 Hybrid
MHA MXFP8 初始化和 buffer 创建；完成 K/V、scale 与 arena 绑定后，再设置
`conv_col_dtype`。因此 recipe 必须一次性规划所有 data、scale 和 checkpoint field。

## 读写路径

- K/V data 和 scale：继承 MXFP8 getter、`set_kv_buffer()` 及融合量化接口；
- KVConv checkpoint：`kvconv_checkpoint_buffers()`；
- HiddenConv checkpoint：`hiddenconv_checkpoint_buffer()`；
- 异构 head：继承 MHA 的 `_layer_row_view()` 与 recipe 的逐层 head count；
- page 清零和 PD：继承 hybrid contract。

## Overlay 与生命周期

MXFP8 data/scale、KVConv 和 HiddenConv 可能占用同一父块中的不同 plane，部分
checkpoint 还会直接 overlay K/V plane。Sleep/wake 时这些 field 共享同一个
`kv_cache` memory-saver region；page reuse 必须先清零，不能只初始化 K/V data 而
遗漏 checkpoint 或 scale bytes。

## 关键不变量

- `head_dim` 必须是 128，且 scale block 固定为 32；
- page token 数必须是 128 的整数倍；
- `set_kv_buffer()` 输入必须符合 MXFP8 data/scale contract；
- ShortConv dtype 环境变量必须与 recipe element size 一致；
- 所有 overlay group 的新 page 都必须按 contract 清零。
