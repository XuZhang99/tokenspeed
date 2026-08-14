# HybridDeepseekV4TokenToKVPool

[子类索引](README.md) · [CacheMemoryPlan](../memory-plan.md)

源码：[`kv_cache/hybrid_deepseek_v4.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py)

Recipe：[`recipes/deepseek_v4.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py)

## 定位与创建条件

该类直接继承 `CachePool`，没有复用普通 MHA/MLA 的 buffer ABI：

```text
CachePool
└── HybridDeepseekV4TokenToKVPool
```

Factory 在 `spec.family == "deepseek_v4"` 时创建，并要求 `spec.pool_options` 提供
`DeepseekV4CacheLayout`。V4 cache 包含 SWA、压缩 KV、compressor state 和 CSA
indexer，多种 block size 无法用一个普通 `kv_buffer` 表达。

## Recipe 几何

DeepSeek V4 recipe 要求：

```text
logical_block_tokens = 256
layer compression ratio ∈ {1, 4, 128}
```

Pool 同时接收：

- `memory_plan`：统一 LCM arena 的物理几何；
- `DeepseekV4CacheLayout`：每层 ratio、shape、block-size 和 indexer 配置；
- `paged_cache_group_specs`：history/state retention 与 rows-per-page；
- `token_capacity`：向 scheduler 发布的容量。

构造会检查 `layer_num == len(layout.layer_ratio)`，以及 `size` 与 plan 的 child
capacity 完全一致。

## 每层字段

所有 layer 都绑定：

```text
layer.<id>.swa                uint8
```

ratio 大于 1 的 layer 再绑定：

```text
layer.<id>.compressed_kv      uint8
layer.<id>.compressor_state   float32
```

ratio 等于 4 的 layer额外绑定：

```text
layer.<id>.indexer_kv         uint8
layer.<id>.indexer_state      float32
```

运行时维护五个与 layer 对齐的 list：

```text
swa_kv_buffer
compressed_kv_buffer
compressor_state_buffer
indexer_kv_buffer
indexer_state_buffer
```

不适用于某 ratio 的条目保存 `None`。`indexer_kv` 被 reshape 为二维 raw-byte
view，其他 field 保留 plan 形状。

## Cache group 关系

主要 group 包括：

```text
v4.swa_kv
v4.c4a.compressed_kv
v4.c128a.compressed_kv
v4.c4a.compressor_state
v4.c128a.compressor_state
v4.c4a.indexer_compressor_state
```

`indexer_kv` 不拥有独立 page group，而是归属于 ratio-4 compressed-KV group，
与其共享 page table 和 page-count budget。Indexer state 则有自己的 state group。

## Block size

Pool 从 runtime group specs 和 layout 计算：

- `swa_block_size`；
- 每层 `compressed_block_sizes`；
- ratio-4 layer 的 `indexer_block_sizes`；
- `compressor_state_block_sizes`；
- `indexer_state_block_sizes`。

History group 与 state group 的 rows-per-page 可以不同，因此调用方不能统一使用
构造参数中的一个 `page_size` 推断所有 buffer 的 row geometry。应使用对应 getter：

```python
get_compressed_block_size(layer_id)
get_indexer_block_size(layer_id)
get_compressor_state_block_size(layer_id)
get_indexer_state_block_size(layer_id)
```

## Buffer accessor

- `get_swa_kv_buffer()`：所有层可用；
- `get_compressed_kv_buffer_2d()`：仅 ratio > 1；
- `get_compressor_state_buffer/view()`：仅 ratio > 1；
- `get_indexer_kv_buffer_2d()`：仅 ratio 4；
- `get_indexer_state_buffer/view()`：仅 ratio 4；
- `swa_capacity_slots`：SWA page 数乘 SWA block rows。

内部 `_require()` 将 `None` 转成带 layer/name 的明确 `ValueError`，调用方不应绕过
这些 getter 直接假定所有 layer 都有相同组件。

## 通用 K/V ABI 的兼容层

为满足只需要“某个 KV buffer”的通用调用方：

```text
get_key_buffer(layer)   → SWA buffer
get_value_buffer(layer) → 同一个 SWA buffer
get_kv_buffer(layer)    → (SWA, SWA)
```

这不表示物理上存在相同的独立 K/V。标准 `set_kv_buffer()` 明确未实现；实际写入
由 DeepSeek V4 attention helper 根据 SWA/compressed/indexer 语义完成。

## Scheduler 诊断

`bind_paged_cache_scheduler()` 保存 scheduler 引用。Debug 日志开启且 rank 0 时，
`maybe_log_paged_cache_group_pages()` 查询所有 state group 的 total/available page，
输出使用率，便于发现 compressor/indexer state 容量不足。

## Page 清零与传输

```python
paged_cache_requires_page_zeroing = True
```

`zero_new_pages()` 按 group 调用 `zero_blocks()`，避免压缩 state 和 indexer live-tail
继承旧请求数据。

旧式 `get_contiguous_buf_infos()` 明确不可用，传输必须走 cache contract。
`get_layerwise_buf_info_offsets()` 仍能为 layer-wise 消费者枚举本层实际存在的
SWA/compressed/state/indexer 条目。

## 显存统计

所有 view 来自同一 arena，`get_kv_size_bytes()` 直接返回：

```text
buffer.nbytes
```

避免按五组 view 累加时重复计算 overlay 或共享区域。

## 关键不变量

- 逻辑父块 token 数必须为 256；
- ratio 只能是 1、4、128；
- layer ratio 数必须等于 `layer_num`；
- 访问 compressed/indexer 组件前必须确认该 layer ratio；
- `indexer_kv` 与 ratio-4 compressed group 共享 page budget；
- 写入必须走 V4 helper，不能调用通用 `set_kv_buffer()`；
- page reuse 必须清零，传输必须使用 cache contract。
