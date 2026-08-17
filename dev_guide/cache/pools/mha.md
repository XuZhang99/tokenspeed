# MHATokenToKVPool

[子类索引](README.md) · [Arena 与字段绑定](../field-binding.md)

源码：[`kv_cache/mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/mha.py)

`MHATokenToKVPool` 是普通 MHA 的计算视图。factory 在 `MHAConfig + family=mha`
时选择它；MXFP8、MSA 和 hybrid MHA 从它继承。

## 字段与绑定

Recipe 为每层声明：

```text
layer.<global_layer>.k  [P, kv_heads_per_rank, head_dim]
layer.<global_layer>.v  [P, kv_heads_per_rank, head_dim]
```

arena 会把首维为 P 的字段折成 token-row view。类的绑定表是：

```python
layer_plane_bindings = {"k": "k_buffer", "v": "v_buffer"}
```

因此 `k_buffer` / `v_buffer` 都是长度为 `layer_num` 的 list；每项通常为：

```text
[group.page_count × P, heads, head_dim]
```

block 0 的 row 也在其中，作为 padding/dummy slot。

## 异构 KV head

`layer_kv_head_counts` 可描述每层 TP 前的真实 head 数，`kv_alloc_head_count` 描述
分配宽度。`_layer_row_view()` 对窄 head 层把相同字节重解释为更多 row：

```text
[rows, allocation_heads, dim] → [more_rows, served_heads, dim]
```

getter 与 writer 必须都经过该函数，否则 loc 与实际 row stride 会不一致。

## 接口

- `get_key_buffer()` / `get_value_buffer()` / `get_kv_buffer()`：必要时先等待
  layer-wise Host L2 加载；
- `set_kv_buffer()`：将输入转为 cache dtype/bit view，再通过
  `tokenspeed-kernel` 的 `store_kv_cache()` scatter 到 `loc`；
- `get_kv_size_bytes()`：按 `data_ptr()` 去重统计，避免 overlay view 重复计数。

`store_dtype` 只用于写入 bit reinterpretation。field 自身 dtype 已由 memory plan
确定；pool 不调用 `field(id, dtype)`。

## 不变量

- `layer_group_ids` 必须覆盖当前 view 的全部 layer；
- plan 必须为 attention layer 声明 K/V，state layer 的相应 list 项可以为 `None`；
- loc、getter 和 writer 必须使用同一 per-layer row view；
- pool 不拥有内存，物理几何从 `self.arena.plan` 读取。
