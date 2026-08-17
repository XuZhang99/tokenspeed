# HybridMHATokenToKVPoolMXFP8

[子类索引](README.md) · [Hybrid MHA](hybrid-mha.md) · [MHA MXFP8](mha-mxfp8.md)

源码：[`kv_cache/hybrid_mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py)

该类通过多重继承组合 history/state view 与 MXFP8 data/scale writer：

```python
class HybridMHATokenToKVPoolMXFP8(
    HybridMHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
):
    ...
```

`layer_plane_bindings` 合并 K/V、conv/ssm、K/V scale 五类属性。plan 决定每层实际
存在的 plane，公共 `_bind_layer_planes()` 一次遍历完成绑定；旧的 MRO
`_create_buffers()` 链已经不存在。

history 层沿用 MXFP8 的 data/scale getter、预量化写入与融合写入；state 层沿用
`get_state_buffers()` / `get_component()`。本类只覆盖 `_layer_page_tokens()`，让 scale
interleave 使用 `arena.kv_page_size`，不按异构 head 再扩张。

它继承 `requires_page_zeroing=True` 和 `zero_new_blocks()`。是否可达由具体 recipe
决定：factory 有 `qwen_gdn + mxfp8` 分支不代表所有 Qwen 配置都能生成所需字段。

## 不变量

- `head_dim % 32 == 0`，scale shape 必须由公共 helper 生成；
- plan 必须同时声明所需 history scale 与 state plane；
- MXFP8 writer 的输入契约与普通 hybrid state accessor 互不混用；
- 新 block 必须按 group 清零。
