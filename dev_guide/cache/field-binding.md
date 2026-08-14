# Arena 与 `field()` 绑定

[上一篇：CacheMemoryPlan](memory-plan.md) · [返回目录](../cache.md) · [下一篇：运行时集成](runtime.md)

## Arena 的分配

物理 buffer 在第一次绑定 field 时由 `_ensure_buffer()` 懒分配：

```python
self.buffer = torch.zeros(
    self.plan.arena_bytes,
    dtype=torch.uint8,
    device=self.device,
)
```

Buffer 始终是一维 `uint8` Tensor。实际显存大小由 `plan.arena_bytes` 决定，
并不是 `size * dtype.itemsize`。使用字节 buffer 可以在同一个 arena 中容纳不同
dtype 的 field，也方便以统一的原始字节布局进行清零和传输。

## `field()`：创建计算视图

代码中实际是单数方法 `field()`，定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/base.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/base.py)。
它负责把 `CacheMemoryPlan` 中某个 field 的纯字节布局绑定成计算代码可以使用的
PyTorch Tensor view：

```text
CacheMemoryPlan 中的字节几何
              │ field(field_id, dtype)
              ▼
共享 arena 上的 typed、strided Tensor view
```

它不会为 field 单独分配或复制显存。所有 view 都以 `CachePool.buffer` 为
storage，只拥有各自的 dtype、shape、offset 和 stride 元数据。

### 绑定流程

`field(field_id, dtype)` 依次执行：

1. 调用 `_ensure_buffer()`，第一次绑定时懒分配 arena；如果当前 pool 有
   `backing_pool`，则复用 backing pool 的 buffer；
2. 查询 `_fields` 注册表；已经绑定的 field 直接返回同一个 Tensor 对象；
3. 通过 `plan.field(field_id)` 取得 `CacheFieldLayout`，未规划的 id 会报错；
4. 检查 `dtype.element_size()` 是否等于 plan 中的 `field.element_size`；
5. 通过 `field.group_id` 查询 group，取得最终 `page_count`；
6. 根据 field 的 byte offset、page stride 和页内 shape 调用 `as_strided()`；
7. 将结果记录到 `_fields[field_id]` 并返回。

同一个 field 只能绑定成一种运行时 dtype。例如第一次绑定为 BF16 后，再用 FP16
请求它，即使两者都是 2 字节，也会因为 dtype 不一致而报错。这样可以避免不同
调用方用不同类型解释同一段存储。第一次绑定时 plan 只校验元素宽度，因为
`CacheMemoryPlan` 是纯整数几何，本身不依赖 PyTorch dtype。

### Shape 和 stride

核心逻辑可以简化为：

```python
view = buffer.view(dtype).as_strided(
    (group.page_count, *field.shape),
    (
        field.page_stride_bytes // field.element_size,
        *inner_contiguous_strides,
    ),
    field_byte_offset // field.element_size,
)
```

最终 Tensor 的 shape 是：

```text
[group.page_count, *field.shape]
```

- 第 0 维表示逻辑 child page，长度由 group 决定；
- 后续维度表示一个 page 内的数据，来自 `CacheFieldLayout.shape`；
- 第 0 维 stride 是 `page_stride_bytes / element_size`；
- 页内维度使用普通 contiguous stride；
- storage offset 由 plane 起点、null page 位置和 field 页内偏移共同决定。

`_field_block_byte_offset()` 使用以下字节几何定位某个 block：

```text
plane.arena_offset_bytes
+ plane.bytes_per_lcm_block
- field.page_stride_bytes
+ block_id * field.page_stride_bytes
+ field.field_offset_bytes
```

其中 `plane.bytes_per_lcm_block - field.page_stride_bytes` 使逻辑 page 0 指向
null parent 的最后一个 child-page slot；逻辑 page 1 从第一个可用 parent 开始。
这样每个 group 都只对外暴露一个统一的 null page 0，即使该 group 的 packing
大于 1。

### 示例

假设 plan 中的 K field 为：

```text
field_id          = layer.0.k
group.page_count  = 1001
field.shape       = (128, 8, 128)
element_size      = 2
page_stride_bytes = 262144
```

调用：

```python
k = pool.field("layer.0.k", torch.bfloat16)
```

会得到：

```text
k.shape = [1001, 128, 8, 128]
k.dtype = torch.bfloat16
```

`k` 没有独立 storage；对 `k` 的读写会直接访问共享 arena 中对应 plane 的字节。
如果另一个 field 的 plan 与它发生 overlay，两个 Tensor view 还可能有意指向同一
物理区域，但使用不同 group 的 page stride 解释这些字节。

### 子类如何使用

例如 MHA pool 会把每一层的 K/V field 转换为 kernel 熟悉的形状：

```python
self.k_buffer = [
    self.field(f"layer.{layer_id}.k", self.store_dtype)
    .view(-1, self.head_num, self.head_dim)
    for layer_id in range(self.layer_num)
]
```

MLA 和 hybrid pool 使用相同入口绑定其他类型的缓存：

```python
self.field("layer.0.latent_kv", dtype)
self.field("layer.7.conv_state", dtype)
self.field("layer.7.recurrent_state", dtype)
```

因此 `field()` 是纯整数布局与运行时计算接口之间的桥梁：plan 决定字节在哪里，
`field()` 决定 PyTorch 和 kernel 如何把这些字节看成 Tensor。
