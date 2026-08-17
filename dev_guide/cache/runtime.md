# Cache 运行时集成

[上一篇：Arena 与字段绑定](field-binding.md) · [返回目录](../cache.md)

## 创建顺序

`registry.create_attn_components()` 的当前顺序是：

1. 根据模型能力选择 `cache_family`；
2. `prepare_cache_setup()` 运行该 family 的统一 recipe；
3. 从合并 spec 派生 target/draft `layer_view()`；
4. `create_cache_arena(spec, ...)` 创建唯一 allocation；
5. `create_cache_pool(target_spec, ..., arena)` 创建 target 计算视图；
6. 如有 draft，再用同一个 arena 和 continuation `field_layer_offset` 创建 draft
   计算视图；
7. backend 配置与 scheduler 都读取 arena 发布的 group contract。

target/draft 不通过 `backing_pool` 互相拥有；共享关系由共同的 `CacheArena` 明确表达。

## `CacheRuntimeContract`

arena 构造时发布：

```python
CacheRuntimeContract(
    prefix_granularity=plan.prefix_granularity,
    num_lcm_blocks=plan.num_lcm_blocks,
    token_capacity=spec.token_capacity,
    group_specs=spec.cache_group_specs,
    group_page_counts={...},
    group_packing={...},
)
```

[`recipes/cache_runtime.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/cache_runtime.py)
验证每个 group：

```text
page_count == num_lcm_blocks × packing + 1
```

并要求 group spec、page-count、packing 三者的 key 集合完全一致。contract 使用
只读 mapping，防止运行时视图修改 scheduler 几何。

## Python → C++ scheduler bridge

[`engine/scheduler_utils.py`](../../python/tokenspeed/runtime/engine/scheduler_utils.py)
的 `pool_to_cache_groups()` 是唯一折叠点：

- row geometry 直接传 `rows_per_page` / `entry_stride_tokens`；
- checkpoint state 在边界折叠成等价的 token span；
- plan 的 `group_page_counts` 与 `group_packing` 填入 C++ `CacheGroupConfig`；
- retention、family、transfer policy 转为绑定 enum。

C++ scheduler 只用 `prefix_granularity` 和每组 `block_granularity` 做 token 逻辑；LCM
packing 只作为 allocator 的物理配置传输。

## 新 block 清零

纯 MHA/MLA pool 的 `requires_page_zeroing` 为 `False`。hybrid state 与 V4 pool 将它
设为 `True`，并实现：

```python
zero_new_blocks(block_ids_by_group)
```

该方法调用 `arena.zero_blocks()`。arena 对每个 `(group, block)` 枚举该 group 的
field，以 `field_page_byte_offset + payload_bytes` 生成 byte ranges，再由 kernel 批量
清零。整个 sleep/wake 修复则调用 `CachePool.clear_kv_buffers()` → `arena.clear()`。

target/draft 可能都被 event loop 遍历；对共享 arena 重复全量 clear 是安全的。

## Host L2 transfer layout

`CachePool.cache_transfer_layout()` 只选择当前 layer window 的字段，再调用
`layout_from_lcm_plan()`。因此：

- target/draft 的计算视图可以各自选择本地字段；
- 字节 offset、stride、dtype 始终来自同一个 memory plan；
- `combine_cache_transfer_layouts()` 可在需要时组合两侧，并保留共享 owner 语义。

## PD contract

PD 从 arena 建立 `CacheTransferContract`：

```text
CacheMemoryPlan + CacheGroupSpec[] + CacheTransferSchema + arena base address
```

`CacheArena.supports_disaggregation` 要求每个 group 都有 transfer policy；
`contract_binding()` 只返回连续的 owner buffer。field 已在 arena 构造期全部物化，
不再需要传输前检查「是否都绑定」或额外携带 dtype map。

详见 [PD 分离](../pd-disaggregation.md)。

## 生命周期不变量

- 一个合并模型只有一个 arena allocation 和一个 runtime contract；
- `CachePool` 的局部 layer id 必须经过 `field_layer_offset` 映射一次且仅一次；
- scheduler/backend/PD/L2 都从 arena/plan 读取几何，不能从 pool 推导副本；
- 需要 state sanitization 的 pool 必须声明 `requires_page_zeroing` 并实现
  `zero_new_blocks()`；
- `clear_kv_buffers()` 是 sleep/wake 全量操作，不等价于每步的新 block 清零。
