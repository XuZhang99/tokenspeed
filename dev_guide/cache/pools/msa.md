# MSATokenToKVPool

[子类索引](README.md) · [父类：MHA](mha.md)

源码：[`kv_cache/msa.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/msa.py)

`MSATokenToKVPool` 为 MiniMax Sparse Attention 增加 key-only side cache。factory 在
`MSAConfig` 时创建。

普通 K/V 完全复用 MHA。只有 `sparse_layer_ids` 中的层由 recipe 声明：

```text
layer.<id>.index_k [P, index_head_dim] model dtype
```

绑定表增加 `index_k → _index_k`。基类先生成带 `None` hole 的逐层 list，本类再只把
存在的项组成：

```python
index_k_buffer: dict[int, Tensor]
```

`get_index_k_buffer()` 会等待可选的 layer-wise load；访问非 sparse layer 抛
`RuntimeError`。与 DSA 不同，MSA pool 不实现 FP8 block-split 量化，index dtype 直接
由 config/plan 决定。

`get_kv_size_bytes()` 把 index 字节计入 key 侧统计。任何传输路径都应从 plan 的
field 集合选择字段，不能只假设普通 MHA 的 K/V 两列。

## 不变量

- `sparse_layer_ids` 与 recipe 实际声明的 index field 必须一致；
- 非 sparse layer 不存在 index view；
- 不要把 DSA 的 raw-byte/block-split 假设用于 MSA；
- 传输 schema 必须覆盖 side-cache field。
