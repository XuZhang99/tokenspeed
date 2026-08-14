# HybridMHATokenToKVPoolMXFP8

[子类索引](README.md) · [Hybrid MHA](hybrid-mha.md) · [MHA MXFP8](mha-mxfp8.md)

源码：[`kv_cache/hybrid_mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py)

## 多重继承定位

该类通过多重继承组合两组能力：

```python
class HybridMHATokenToKVPoolMXFP8(
    HybridMHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
):
```

```text
HybridMHATokenToKVPool → history/state 混合绑定、清零、contract
MHATokenToKVPoolMXFP8 → FP8 data、UE8M0 scale、MXFP8 写入
```

它也是 `HybridInklingTokenToKVPoolMXFP8` 的能力基类。

## MRO 下的 buffer 创建

本类覆盖 `_create_buffers()`：

1. 检查 `head_dim % 32 == 0`；
2. 将 `store_dtype` 设置为 `float8_e4m3fn`；
3. 调用 `super()._create_buffers()`，按 MRO 进入 Hybrid MHA 的 history/state
   `_bind_buffers()`；
4. 再绑定每层 `k_scale` 和 `v_scale` field。

最终同时存在：

```text
attention/history:
  layer.<id>.k
  layer.<id>.v
  layer.<id>.k_scale
  layer.<id>.v_scale

state:
  模型 recipe 规划的 recurrent-state fields
```

具体 plan 必须包含构造过程请求的全部 scale field，否则 `field()` 会立即失败。

## MXFP8 行为

K/V data、scale getter、预量化写入和融合量化写入继承自
`MHATokenToKVPoolMXFP8`。本类覆盖 `_layer_page_tokens()`，固定返回
`self.page_size`，使 hybrid recipe 的 page geometry 成为 scale interleave 的权威
来源，而不是按 head-width 再扩展 page token 数。

Scale 的 block size 仍是 32，data 仍是 `float8_e4m3fn`，scale 仍是
`float8_e8m0fnu`。

## Hybrid 行为

以下能力来自 `HybridMHATokenToKVPool`：

- layer/group/state 映射；
- recurrent state accessor；
- `paged_cache_requires_page_zeroing=True`；
- `zero_new_pages()` 和全 arena clear；
- 禁用旧式 contiguous transfer，要求 plan-based PD contract。

## Factory 与当前可达性

Factory 在 `qwen_gdn + kv_cache_mxfp8` 时具有选择该类的分支，但当前 Qwen cache
recipe 在准备阶段会明确报错：Qwen 尚不支持该 interleaved scale layout。因此，
当前主要实际用途是作为 Inkling MXFP8 组合类的父类。

这个区别很重要：类存在、factory 有分支，并不意味着每个 recipe 都能成功生成
它所需的 scale fields。

## 关键不变量

- `head_dim` 必须能被 32 整除；
- plan 必须同时覆盖 hybrid state 与 K/V scale fields；
- writer 必须提供 MXFP8 data 和 scale，或成功走融合量化；
- page reuse 必须清零；
- transfer 必须使用 cache contract；
- 使用前应先确认对应 model recipe 没有禁用 MXFP8。
