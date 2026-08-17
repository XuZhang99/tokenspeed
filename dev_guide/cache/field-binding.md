# CacheArena 与字段绑定

[上一篇：CacheMemoryPlan](memory-plan.md) · [返回目录](../cache.md) · [下一篇：运行时集成](runtime.md)

字段物化由
[`kv_cache/arena.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)
负责，不再由 `CachePool.field(field_id, dtype)` 延迟完成。

## 一次性分配

`CacheArena.__init__()` 在 memory-saver 的 `kv_cache` region 中立即分配：

```python
self.buffer = torch.zeros(
    plan.arena_bytes,
    dtype=torch.uint8,
    device=device,
)
```

一维 `uint8` owner buffer 可以同时容纳不同 dtype，且 `nbytes` 与 plan 的字节几何
严格一致。arena 随即遍历 `plan.fields`，将每个 field 物化到 `_fields`；不存在
「首次 getter 才分配」或「target pool 先绑定，draft 再复用注册表」的状态。

## `_bind()` 的 view 几何

对一个 `CacheFieldLayout`，arena 从 `field.dtype` 得到 torch dtype，并创建：

```text
[group.page_count, *field.shape]
```

第 0 维 stride 是：

```text
field.page_stride_bytes / field.element_size
```

页内维度使用 contiguous stride，storage offset 来自
`plan.field_page_byte_offset(field_id, 0)`。所有 view 与 owner buffer 零拷贝共享
storage。

如果 `field.shape[0] == plan.prefix_granularity`，这个 field 表示逐 token 条目，arena
把 page 轴与首个 shape 轴折叠：

```text
[page_count, P, ...] → [page_count × P, ...]
```

recurrent state、block-scaled scale plane、V4 raw-byte plane 等首维不是 `P` 的字段
保留 page-major shape。是否折叠由 plan 决定，pool 不再重复判断 kernel 几何。

## null block 的偏移

`CacheMemoryPlan.field_page_byte_offset()` 使用：

```text
plane.arena_offset_bytes
+ plane.bytes_per_lcm_block
- field.page_stride_bytes
+ page_id × field.page_stride_bytes
+ field.field_offset_bytes
```

因此逻辑 block 0 指向 null parent 的最后一个 child slot；逻辑 block 1 才进入第一
个可用 parent。不同 packing 的 group 都只暴露一个 null block 0。

## `field()` 是只读查表接口

```python
tensor = arena.field("layer.0.k")
```

`field()` 不接受 dtype，不创建新 view；它只返回构造期已经物化的 Tensor。未知 id
抛 `ValueError`。`field_ids()` 可用于检查 plan 的完整字段集合。

## Pool 如何取得逐层 buffer

子类声明：

```python
layer_plane_bindings = {
    "k": "k_buffer",
    "v": "v_buffer",
}
```

`CachePool._bind_layer_planes()` 解析 `layer.<global_id>.<plane>`，只选择当前
`[field_layer_offset, field_layer_offset + layer_num)` window 内的字段，并生成长度为
`layer_num` 的 list。未在某层规划的 plane 为 `None`，例如 hybrid state layer 没有
K/V，而 ratio-1 的 V4 layer 没有 compressed/state plane。

Inkling 的按需 accessor 也是查表：

```python
self.arena.field(f"layer.{global_layer}.kvconv_k")
```

它并不延迟物化或选择 dtype。

## dtype 的单一来源

dtype 在 recipe 创建 `CacheFieldSpec` 时确定，沿
`CacheFieldLayout → CacheMemoryPlan → CacheArena` 传递。plan 保持 torch-free，使用
字符串名；arena 才调用 `getattr(torch, field.dtype)`。

`CachePool.store_dtype` 只描述写入时如何重新解释输入（例如 scatter 不支持 FP8
时转成 `uint8` bit view），不决定 arena field 的 dtype。新增字段时必须在 recipe
中把 dtype 与 shape 一起声明。
