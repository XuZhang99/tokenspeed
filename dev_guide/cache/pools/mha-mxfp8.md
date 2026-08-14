# MHATokenToKVPoolMXFP8

[子类索引](README.md) · [父类：MHATokenToKVPool](mha.md)

源码：[`kv_cache/mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/mha.py)

## 定位与创建条件

该类继承 `MHATokenToKVPool`，为 MHA 提供 block-scaled MXFP8 K/V 存储：

```text
CachePool
└── MHATokenToKVPool
    └── MHATokenToKVPoolMXFP8
```

Factory 在 `MHAConfig`、`family == "mha"` 且 `kv_cache_mxfp8=True` 时选择它。

## 数据和 scale 字段

每层需要四个 field：

```text
layer.<id>.k          float8_e4m3fn data
layer.<id>.v          float8_e4m3fn data
layer.<id>.k_scale    float8_e8m0fnu scale
layer.<id>.v_scale    float8_e8m0fnu scale
```

`MXFP8_SCALE_BLOCK_SIZE = 32`，即 head-dim 每 32 个值共享一个 UE8M0 scale，
所以 `head_dim` 必须能被 32 整除。

`_create_buffers()` 先把 `store_dtype` 设置为 `float8_e4m3fn`，复用父类创建 K/V
view，再绑定 K/V scale view。

## Scale 布局

令：

```text
scale_dim = head_dim / 32
```

普通 page size 为 128 时，scale 按 FA4 block-scaled kernel 需要的 interleaved
布局保存，单页逻辑形状为：

```text
[heads, 32, scale_dim, scale_dim]
```

例如 `head_dim=128` 时 `scale_dim=4`。其他 page size 使用 flat slot 布局：

```text
[page_size, heads, scale_dim]
```

异构 KV head 模式下，`_layer_scale_view()` 会结合 `_layer_page_tokens()` 将统一
物理 scale field 解释成当前层的 head/row 组合：

```text
[num_page_ids, heads_l, page_tokens/128, 32, scale_dim, scale_dim]
```

## 写入路径

`set_kv_buffer()` 与 BF16 父类不同：

- 输入 K/V 必须已经是 `float8_e4m3fn`；
- `k_scale` 和 `v_scale` 必须提供；
- K/V 以 `uint8` bit view 交给 `store_kv_cache()`，避开 Triton FP8 mask-fill
  限制；
- scale 使用 `store_sf_interleaved()` 写入 interleaved 布局，或在 flat 布局下
  直接按 `loc` 赋值。

`layer_id_override` 允许调用方显式指定写入层，供组合类和特殊 backend 复用。

## 融合量化写入

`quantize_and_set_kv_buffer()` 尝试用一次 kernel 完成：

```text
BF16/FP16 K,V
    ↓ MXFP8 quantize
FP8 data + UE8M0 scale
    ↓ scatter
data field + scale field
```

只有满足以下条件时走融合路径：

- scale 是 interleaved page 布局；
- `head_dim == 128`。

否则返回 `False`，由调用方退回“先量化、再调用 `set_kv_buffer()`”的分离路径。

## 读取和统计

K/V getter 沿用父类，scale 通过：

```python
get_kv_scale_buffer(layer_id) -> (k_scale, v_scale)
```

暴露给 block-scaled attention kernel。`get_kv_size_bytes()` 在父类 K/V 字节基础
上，再按 data pointer 去重统计 scale field，避免 alias plane 被重复计算。

## 与普通 MHA 的主要差异

| 项目 | 普通 MHA | MXFP8 MHA |
| --- | --- | --- |
| K/V dtype | BF16/FP16/普通 cache dtype | `float8_e4m3fn` |
| scale | 可选 per-tensor 参数 | 必需的 per-token block scale field |
| 写入输入 | 可由 pool 转 dtype | 必须预量化，或走融合量化接口 |
| 附加 field | 无 | `k_scale`、`v_scale` |
| 约束 | 常规 head geometry | `head_dim % 32 == 0` |

## 注意事项

- 不要把 BF16 Tensor 直接传给 `set_kv_buffer()`；
- scale shape 必须与 data 的 page/head 解释一致；
- 异构 head 场景应通过 `_layer_page_tokens()` 计算每个 page id 对应的 token row；
- 融合接口返回 `False` 是正常 fallback 信号，不代表写入已完成。
