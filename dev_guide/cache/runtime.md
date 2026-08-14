# CachePool 运行时集成

[上一篇：Arena 与 field()](field-binding.md) · [返回目录](../cache.md)

## 构造参数

几个容易混淆的参数如下：

- `size`：逻辑 pool size。基类主要将其作为元数据，以及未显式传入
  `token_capacity` 时的缺省值；它不是 arena 的字节数。
- `dtype`：KV 的逻辑 dtype。对于部分 FP8 dtype，底层 `store_dtype` 会改为
  `uint8`，以规避 Torch 写入算子的限制。
- `device`：arena 所在设备。
- `page_size`：一个逻辑 cache block 对应的 token 数，通常等于
  `plan.logical_block_tokens`。
- `rank`：当前并行 rank 的元数据。
- `memory_plan`：物理布局和显存分配的权威来源。
- `paged_cache_group_specs`：各 group 的 family、retention、rows-per-page 等调度语义。
- `token_capacity`：向 scheduler 公布的有效 token 容量。
- `backing_pool`：让当前对象作为另一个 pool 的共享计算视图。
- `field_layer_offset`：把当前 view 的局部 layer id 映射到合并 plan 中的全局 layer。
- `pd_disaggregation_enabled`：是否允许发布 PD 分离传输契约。

## Scheduler 运行时契约

`_publish_runtime_contract()` 将 recipe 给出的逻辑 group spec 与 plan 中的真实
packing/page count 对齐，生成 `PagedCacheRuntimeContract`：

```python
PagedCacheRuntimeContract(
    block_size=page_size,
    num_lcm_blocks=plan.num_lcm_blocks,
    token_capacity=token_capacity,
    group_specs=aligned_group_specs,
    group_page_counts=group_page_counts,
)
```

Scheduler 通过该契约获得：

- 缓存组集合；
- 每组的 page 数；
- 每个 LCM 父块包含多少个该组的子 page；
- group 的 history/state、retention 和 sliding-window 等语义；
- 可调度的有效 token 容量。

`CachePool` 因此是物理内存布局与调度器逻辑分页之间的契约边界。

## 共享 backing pool

Target 和异构 draft 可以共享同一个 arena。带 `backing_pool` 构造的 view：

- 不分配新 buffer；
- 复用 backing pool 的 `buffer` 和 `_fields` 注册表；
- 继承 backing pool 的 `runtime_contract`；
- 通过 `field_layer_offset` 绑定 draft 对应的 continuation layer。

构造顺序必须是 target-first：backing pool 必须已经绑定 field 并完成 arena
分配。共享 view 也不能重复发布 `paged_cache_group_specs`，只能继承 backing pool
已经发布的运行时契约。

## 清零与传输

`CachePool` 提供两种清零方式：

- `zero_blocks()`：只清零指定 group 的指定 CacheBlock，按 plan 计算原始字节区间；
- `clear_kv_buffers()`：清空整个 arena，主要用于 sleep/wake 后修复重新映射的存储。

纯 attention pool 默认不要求 page 重用时清零。KV 与 recurrent state 共享物理
页面的 hybrid pool 会设置：

```python
paged_cache_requires_page_zeroing = True
```

这样可以避免旧 state 的尾部数据污染新请求。

传输相关接口包括：

- `pd_contract()`：生成 prefill/decode 分离所需的跨节点原始 slab 契约；
- `get_pd_cache_contract()`：在启用 PD disaggregation 后返回上述契约；
- `cache_transfer_layout()`：生成 Host L2/offload 所需的字节布局。

`pd_contract()` 要求 plan 中的全部 field 已经绑定，因为构建传输契约时需要知道
每个 field 的实际运行时 dtype。

## 创建与使用流程

完整生命周期可以概括为：

1. Cache recipe 根据模型结构和显存预算生成 `CachePoolSpec` 与
   `CacheMemoryPlan`；
2. `create_cache_pool()` 根据 family/config 选择具体 pool 子类；
3. `CachePool.__init__()` 保存 plan，并发布 scheduler runtime contract；
4. 子类在初始化时调用 `field()`，首次调用触发 arena 分配并建立所有计算 view；
5. Scheduler 根据 runtime contract 分配 page id；
6. Attention backend 通过具体子类的 `set_kv_buffer()` 写入，通过
   `get_*_buffer()` 取得对应 view；
7. PD、Host L2 和 sleep/wake 路径复用同一个 plan 描述进行传输或清零。
