# HybridDeepseekV4TokenToKVPool

[子类索引](README.md) · [CacheMemoryPlan](../memory-plan.md)

源码：[`kv_cache/hybrid_deepseek_v4.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_deepseek_v4.py)

几何：[`attention/deepseek_v4_geometry.py`](../../../python/tokenspeed/runtime/layers/attention/deepseek_v4_geometry.py)

Recipe：[`recipes/deepseek_v4.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v4.py)

V4 pool 直接继承 `CachePool`，为 SWA、compressed KV、compressor state 和 CSA
indexer 提供专用 view。factory 要求 `DeepseekV4PoolOptions` 携带当前 layer window 的
`DeepseekV4CacheLayout`。

## 几何来源

V4 kernel 几何不放在 recipe 下。compression ratio、field byte formula、group-id
vocabulary 和 `DEEPSEEK_V4_PAGE_SIZE` 都由 `deepseek_v4_geometry.py` 唯一定义；recipe
只是把它翻译成 group/field declaration。

V4 支持的 layer ratio 为 1、4、128。默认 P 来自 V4 kernel page 常量，但 runtime
允许任何正的 kernel-page 倍数；不能再写死 `prefix_granularity == 256`。

## 每层 plane

所有 layer：

```text
layer.<id>.swa
```

ratio > 1：

```text
compressed_kv
compressor_state
```

ratio == 4：

```text
indexer_kv
indexer_state
```

`indexer_kv` 与 ratio-4 compressed group 共享 block table/page budget，而
`indexer_state` 有独立 state group。`layer_plane_bindings` 创建五个 holey list，plan
中不存在的 plane 为 `None`。

## Block geometry 与接口

不同 group 可有不同 row geometry。pool 从 `arena.cache_group_specs` 和 V4 layout
得到 SWA、compressed、compressor-state、indexer 的 block size，并提供：

- `get_swa_kv_buffer()`；
- `get_compressed_kv_buffer_2d()` / `get_compressed_block_size()`；
- `get_compressor_state_buffer()` / 对应 block size；
- `get_indexer_kv_buffer_2d()` / `get_indexer_block_size()`；
- `get_indexer_state_buffer()` / 对应 block size。

访问不适用于该 ratio 的组件会显式 `ValueError`。通用 K/V getter 都返回 SWA view；
`set_kv_buffer()` 不实现，写入必须走 V4 attention helper。

`DeepseekV4CacheMetadata` 维护各组 block table/base offset，并把 logical token position
映射成 compressed/indexer slot；kernel page 几何来自专用 layout，不能从 P 推导。

## Scheduler、清零和统计

`bind_cache_scheduler()` 保存诊断引用；`maybe_log_cache_group_pages()` 在 debug/rank 0
下报告 state group 的 total/available blocks。

V4 设置 `requires_page_zeroing=True`，`zero_new_blocks()` 交给 arena。显存统计直接
返回 `arena.buffer.nbytes`，避免多组 overlay view 重复计数。

## 不变量

- V4 byte/kernel geometry 只能从 `deepseek_v4_geometry.py` 读取；
- P 必须是 kernel page 的正倍数，而不是固定常量；
- `layer_num` 必须匹配当前 `DeepseekV4CacheLayout.layer_ratio` window；
- ratio 决定组件是否存在，调用方必须使用 accessor；
- 写入走 V4 helper，复用 block 前必须清零。
