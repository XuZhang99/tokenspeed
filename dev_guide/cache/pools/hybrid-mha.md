# HybridMHATokenToKVPool

[子类索引](README.md) · [父类：MHA](mha.md)

源码：[`kv_cache/hybrid_mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py)

该类让 MHA history 与 recurrent state 成为同一 `CacheArena` 上的不同 group view。
factory 对 `MHAConfig + family=qwen_gdn` 选择它。

## 按 plan 绑定

绑定表把两类 plane 合在一起：

```python
{
    "k": "k_buffer",
    "v": "v_buffer",
    "conv": "_conv_state",
    "ssm": "_ssm_state",
}
```

Attention layer 的 plan 只有 K/V，state layer 的 plan 只有 conv/ssm；
`_bind_layer_planes()` 因而自然生成 holey list，再为 state layer 建
`_state_buffers_by_layer`。dtype 与 shape 全部由 plan 决定，不再有
`state_field_dtypes` map 或 pool 侧 stride 重校验。

## 接口

- MHA layer：继承 `get_*_buffer()` 与 `set_kv_buffer()`；
- state layer：`get_state_buffers()` 返回 `(conv, ssm)`；
- `get_component(..., "conv_state" | "recurrent_state")` 提供统一 state ABI；
- `group_id_for_layer()` 返回 recipe 给出的逐层 group；
- `state_slabs` 按 state layer 顺序返回 pair；
- `num_lcm_blocks` 从 `arena.plan` 读取。

错误地对 state layer 访问 K/V，或对 attention layer访问 state，都会显式失败。

## 清零与传输

该类设置 `requires_page_zeroing=True`。`zero_new_blocks()` 把 scheduler 返回的
`{group_id: block_ids}` 交给 `arena.zero_blocks()`，避免复用 parent 时保留旧 recurrent
tail。sleep/wake 全量清零仍走 `arena.clear()`。

PD contract 与 Host L2 layout 都从 arena plan 生成；不存在 hybrid 专用的
`get_pd_cache_contract()` 或旧 contiguous buffer ABI。

## 不变量

- `layer_types` 与 `layer_group_ids` 必须覆盖整个局部 view；
- state/attention consumer 必须使用各自 accessor；
- 复用的新 state block 必须在计算前清零；
- group、dtype、shape、stride 只能由 recipe/plan 提供。
