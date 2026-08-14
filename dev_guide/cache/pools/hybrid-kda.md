# HybridKDATokenToKVPool

[子类索引](README.md) · [父类：MLATokenToKVPool](mla.md)

源码：[`kv_cache/hybrid_kda.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_kda.py)

Recipe：[`recipes/kimi_k3.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py)

## 定位与创建条件

该类让 Kimi-K3 的 MLA latent history 与 KDA recurrent state 共享一个 arena：

```text
CachePool
└── MLATokenToKVPool
    └── HybridKDATokenToKVPool
```

Factory 在 `MLAConfig + spec.family == "kimi_k3"` 时创建。

## Layer 和 group 组织

构造时要求 `layer_types`、`layer_group_ids` 都与 `layer_num` 等长。典型 Kimi-K3
recipe 将 full-attention MLA layer 放入 `full_attention` group，将 KDA layer
划分到多个 `linear_attention_*` state group；具体 group 数和 packing 由 recipe
决定，pool 不硬编码这些容量参数。

内部保存：

```text
_group_ids_by_layer
_state_field_dtypes
_state_buffers_by_layer
```

## 字段绑定

MLA layer 绑定：

```text
layer.<id>.latent_kv
```

并展平为：

```text
[group.page_count * page_size, 1, kv_cache_dim]
```

KDA state layer 绑定：

```text
layer.<id>.conv_state
layer.<id>.recurrent_state
```

对应 dtype 必须由 `state_field_dtypes` 提供。`kv_buffer` 是长度为 `layer_num` 的
list：MLA layer 为 Tensor，KDA layer 为 `None`；state Tensor 保存在 dict 中。

## 构造期校验

`_bind_buffers()` 检查：

1. 不允许 `quant_method == "per_token_head"`；
2. `plan.logical_block_tokens == page_size`；
3. `size == num_lcm_blocks * max_packing * page_size`；
4. 每个 state field 都有 runtime dtype；
5. MLA latent child page 之间没有 padding，能够安全展平为 token rows。

KDA state field 设置 `exact_page_stride=False`，可利用 MLA plane slack；kernel 必须
使用运行时 stride。

## Component 接口

```python
get_component(layer_id, component_name)
```

支持：

| component | 合法 layer | 返回值 |
| --- | --- | --- |
| `latent_kv` | MLA layer | 未展平的 planned field view |
| `conv_state` | KDA layer | conv state Tensor |
| `recurrent_state` | KDA layer | recurrent state Tensor |

错误 layer 类型或未知 component 会报错。`get_state_buffers(layer_id)` 返回 KDA
layer 的 `(conv_state, recurrent_state)`；`group_id_for_layer()` 返回 scheduler
group id。所有 component 访问遵循 layer-wise load 等待。

## 继承的 MLA 接口

MLA layer 继续使用父类：

- `get_key_buffer()` / `get_value_buffer()`；
- `set_mla_kv_buffer()` / `get_mla_kv_buffer()`；
- latent/nope/rope 联合存储。

对 KDA layer 调用这些普通 MLA getter 会因 `kv_buffer[layer] is None` 而失败，必须
改用 state/component 接口。

## Page 清零

```python
paged_cache_requires_page_zeroing = True
```

MLA history 与 KDA state 在统一 LCM parent 中 overlay。新 page 由
`zero_new_pages()` 按 group 清零；`clear_kv_buffers()` 清空整个 arena。未清零的
recurrent tail 会把上一个请求的状态带入新请求。

## 统计与传输

`get_kv_size_bytes()` 返回整个共享 `buffer.nbytes`，而不是分别累加 latent/state
view，以免 overlay 重复计数。

旧式 `get_contiguous_buf_infos()` 和 `get_layerwise_buf_info_offsets()` 都不适用于该
混合布局，会抛出 `RuntimeError`。PD/Host 传输必须依赖 plan-based cache contract
和 transfer layout。

## 关键不变量

- KDA 不支持 MLA `per_token_head` 量化；
- layer type、group id、state dtype 三组元数据必须一致；
- latent page 必须连续，state field 可以带 runtime stride；
- MLA 和 KDA layer 必须使用不同 accessor；
- 新 state page 必须清零；
- 传输不能退回只理解普通 MLA buffer 的旧 ABI。
