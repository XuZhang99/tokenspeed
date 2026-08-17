# DSATokenToKVPool

[子类索引](README.md) · [父类：MLA](mla.md)

源码：[`kv_cache/dsa.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py)

`DSATokenToKVPool` 在 MLA latent cache 上增加每层 `index_k`。factory 对
`DSAConfig` 优先选择该类。

## index-K 物理格式

Recipe 声明：

```text
layer.<id>.index_k [P, dsa_index_k_row_bytes(index_head_dim)] uint8
```

每个 page 的 raw bytes 采用 block-split 格式：先连续存所有 FP8 key，再连续存
float32 scale；它不是逐 token 的 `[FP8 | scale]` 交错行。

`set_index_k_buffer()` 对输入做 `group_size=128` 的 token-group FP8 量化，然后调用
`index_k_block_split_scatter()`。kernel 使用 `arena.kv_page_size` 把 flat loc 分解为
page 与 page 内 offset。

`get_index_k_buffer()` 返回 raw planned field；需要连续 `(values, scales)` 的 consumer
必须使用与 block-split 格式匹配的 gather 路径，不能把 raw row 当交错结构读取。

`get_kv_size_bytes()` 在父类 MLA 统计上增加 index-K view 的字节数。

## 不变量

- `index_head_dim` 按 128 分组；
- index field dtype 固定为 `uint8` raw storage；
- scatter/gather 必须共享相同 P、head dim 与 group size；
- latent/nope/rope 行为仍以 MLA 父类为准。
