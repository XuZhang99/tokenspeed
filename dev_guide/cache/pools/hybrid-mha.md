# HybridMHATokenToKVPool

[子类索引](README.md) · [父类：MHATokenToKVPool](mha.md)

源码：[`kv_cache/hybrid_mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py)

## 定位与创建条件

该类让 MHA history 与 recurrent state 共享一个 LCM arena：

```text
CachePool
└── MHATokenToKVPool
    └── HybridMHATokenToKVPool
```

Factory 对 `MHAConfig + qwen_gdn` 选择它。当前 Qwen recipe 会拒绝
`kv_cache_mxfp8`，所以实际 Qwen GDN 使用本类的非 MXFP8 版本。

## Layer 分类

构造时接收与 `layer_num` 等长的：

- `layer_types`：区分 attention layer 与 `STATE_LAYER_TYPES`；
- `layer_group_ids`：每层所属的 scheduler cache group；
- `state_field_dtypes`：state field id 到运行时 dtype 的映射。

内部保存：

```text
_group_ids_by_layer
_state_layer_ids
_state_buffers_by_layer
```

## 字段绑定

Attention layer 绑定：

```text
layer.<id>.k
layer.<id>.v
```

State layer 绑定：

```text
layer.<id>.conv
layer.<id>.ssm
```

运行时结构为：

```text
k_buffer[layer] / v_buffer[layer]
  attention layer → Tensor
  state layer     → None

_state_buffers_by_layer[layer]
  state layer → (conv, ssm)
```

因此不能对 state layer 调用父类 K/V getter；父类会检测到 `None` 并报错。

## 构造期几何校验

`_bind_buffers()` 会 fail fast：

1. `plan.logical_block_tokens` 必须等于 pool `page_size`；
2. `size` 必须等于
   `num_lcm_blocks * max_packing * page_size`；
3. 每个 state field 必须在 `state_field_dtypes` 中有 dtype；
4. SSM page row 必须连续，满足 GDN decode ABI；
5. Attention K/V page payload 必须连续，才能展平为 token row。

这些检查把 recipe、scheduler capacity 和 kernel stride contract 绑在一起，避免
错误 plan 到运行时才产生静默数据错位。

## State 访问接口

- `state_slabs`：按 state layer 顺序返回全部 `(conv, ssm)`；
- `group_id_for_layer(layer_id)`：返回物理 group id；
- `get_state_buffers(layer_id)`：只接受 state layer；
- `get_component(layer_id, "conv_state")`：返回 conv；
- `get_component(layer_id, "recurrent_state")`：返回 SSM。

`get_component()` 同样等待 layer-wise load tracker。对 attention layer请求 state，
或使用未知 component name，都会明确报错。

## Attention 读写

Attention layer 继续复用父类：

- `get_key_buffer()` / `get_value_buffer()`；
- `_layer_row_view()` 的异构 head 解释；
- `set_kv_buffer()` 的 K/V scatter。

Hybrid 类只替换 `_create_buffers()`，使 state layer 跳过 K/V 绑定并改绑 state。

## Page 清零

```python
paged_cache_requires_page_zeroing = True
```

原因是 history 与 recurrent state 共享物理 parent。新请求复用 page 时，旧 state
尾部不能依赖 attention overwrite 自动覆盖，必须调用：

```python
zero_new_pages(new_page_ids_by_group)
```

它最终通过 `CachePool.zero_blocks()` 按 group 清零所有对应 field。完整
`clear_kv_buffers()` 会清空整个共享 arena。

## 传输

旧式 `get_contiguous_buf_infos()` 不能表达 history/state overlay，因此本类直接
抛出 `RuntimeError`。PD 必须使用 `get_pd_cache_contract()`，由 memory plan、group
spec 和全部 field dtype 共同描述原始 slab。

## 关键不变量

- layer types、group ids 必须覆盖每层；
- state dtype map 必须完整；
- state 与 attention layer 必须走各自 accessor；
- 复用的新 page 必须先清零；
- hybrid transfer 必须走 plan-based cache contract。
