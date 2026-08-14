# DSATokenToKVPool

[子类索引](README.md) · [父类：MLATokenToKVPool](mla.md)

源码：[`kv_cache/dsa.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py)

## 定位与创建条件

`DSATokenToKVPool` 继承 `MLATokenToKVPool`，在 MLA latent cache 之外增加 DSA
稀疏检索使用的 index-K：

```text
CachePool
└── MLATokenToKVPool
    └── DSATokenToKVPool
```

Factory 在 `config` 是 `DSAConfig` 时优先选择该类，并从 config 传入
`index_head_dim`。MLA 的 latent/nope/rope 读写全部由父类提供。

## 新增字段

每层额外绑定：

```text
layer.<id>.index_k
```

运行时底层 view 为：

```text
index_k_buffer[layer].shape = [num_slots, index_k_row_bytes]
dtype = uint8
```

每个 token 的逻辑行包含：

```text
index_head_dim 个 FP8 值
+ index_head_dim / 128 个 FP32 scale
```

因此：

```text
index_k_row_bytes =
    index_head_dim
    + (index_head_dim / 128) * sizeof(float32)
```

## 为什么采用 page 内 block-split

DeepGEMM 的 `fp8_paged_mqa_logits` 期望一个 page 内先连续存放所有 FP8 key，
然后连续存放所有 scale：

```text
一个 page
├── [page_size, index_head_dim] FP8 values
└── [page_size, num_groups]     FP32 scales
```

而不是逐 token 交错成：

```text
token 0 [FP8 | scale], token 1 [FP8 | scale], ...
```

`_index_k_block_views()` 在同一个 `uint8` storage 上建立两个 `as_strided` view：

```text
fp8_view:   [num_pages, page_size, index_head_dim]
scale_view: [num_pages, page_size, index_head_dim / 128]
```

二者不复制数据，写入会直接落到原始 `index_k_buffer`。

## 写入路径

`set_index_k_buffer(layer_id, loc, index_k)`：

1. 将输入转换为 `model_dtype`；
2. reshape 为 `[num_tokens, index_head_dim]`；
3. 以 `group_size=128` 做 per-token-group FP8 量化；
4. 得到 `float8_e4m3fn` values 和 FP32 scale；
5. 调用 `index_k_block_split_scatter()`，在 kernel 内由 flat `loc` 算出
   `(page, offset_in_page)` 并写入 block-split 布局。

量化 granularity 和 scale encoding 是固定 contract，不能由调用方任意改变。

## Gather 路径

非分页 prefill scoring kernel 需要连续的 `(k_fp8, k_scale)`。因此：

```python
gather_index_k(layer_id, slots)
```

先把 flat slot 转为：

```text
page = slot // page_size
offset = slot % page_size
```

再从两个 block-split view 收集：

```text
k_fp8:  [num_slots, index_head_dim]
k_scale:[num_slots, index_head_dim / 128]
```

直接按原始 `[slot, row_bytes]` 索引会把 page 内分开的 value/scale 错当成逐 token
交错布局，因此必须使用该接口。

## 与 MLA 接口的组合

DSA layer 同时具有：

- 父类的 `latent_kv`，或量化模式下的 latent/scale/rope 三元组；
- 本类的 `index_k`。

`has_index_k_buffer()` 固定返回 `True`，backend 可据此选择 DSA scoring 路径。
`get_index_k_buffer()` 也遵循 layer-wise load tracker 的同步约定。

## 统计与传输

- `get_kv_size_bytes()` 在父类 latent cache 字节上加上全部 index-K；
- `get_contiguous_buf_infos()` 在父类 buffer 列表后追加每层 index-K 的 pointer、
  总字节数和 page item bytes；
- `get_layerwise_buf_info_offsets()` 为每层追加 index-K 条目，父类普通模式从
  `layer_num` 后开始，per-token-head 模式从 `3 * layer_num` 后开始。

## 关键不变量

- `index_head_dim` 应按 128 分组；
- index-K 的 runtime dtype 固定为 `uint8` raw storage；
- page 内格式是 block-split，不是 token-interleaved；
- scatter 与 gather 必须使用相同的 `page_size`、head dim 和 group size；
- DSA 的 latent cache 行为仍以父类 MLA 文档为准。
