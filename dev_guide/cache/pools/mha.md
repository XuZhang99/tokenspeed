# MHATokenToKVPool

[子类索引](README.md) · [Arena 与 `field()`](../field-binding.md)

源码：[`kv_cache/mha.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/mha.py)

## 定位与创建条件

`MHATokenToKVPool` 直接继承 `CachePool`，为普通 Multi-Head Attention 提供按层
K/V 读写接口：

```text
CachePool
└── MHATokenToKVPool
```

`create_cache_pool()` 在 `config` 是 `MHAConfig`、`spec.family == "mha"` 且未启用
`kv_cache_mxfp8` 时创建该类。普通 MHA recipe 会把每个 group 的 packing 固定为
1，因此：

```text
group.page_count = 1 + plan.num_lcm_blocks
有效 token 容量 = (group.page_count - 1) * page_size
```

## 初始化数据

除基类参数外，该类保存：

- `head_num`：当前 rank 的 KV head 物理分配宽度；
- `head_dim`：单个 head 的维度；
- `layer_num`：当前计算 view 覆盖的层数；
- `layer_cache_group_ids`：每层所属的物理 cache group；
- `layer_kv_head_counts`：可选的逐层 KV head 数；
- `kv_alloc_head_count`：异构 head 布局的全局分配宽度；
- `field_layer_offset`：target/draft 共享 plan 时的 layer 偏移。

`layer_group_ids` 必须与 `layer_num` 等长，否则构造立即失败。Buffer 创建被放在
`TorchMemorySaverAdapter` 的 `kv_cache` region 中，sleep 时不做 CPU backup，wake
后由分页写入和清零逻辑恢复。

## 字段和 shape

每层绑定两个 field：

```text
layer.<global_layer_id>.k
layer.<global_layer_id>.v
```

`_field_layer_id(local_id)` 返回：

```text
field_layer_offset + local_id
```

所以异构 draft 的本地 layer 0 可以绑定到合并 plan 的 continuation layer。

`field()` 最初返回：

```text
[group.page_count, page_size, head_num, head_dim]
```

随后 `_create_kv_buffers()` 展平 page 和 page 内 token：

```text
k_buffer[layer].shape =
v_buffer[layer].shape =
[group.page_count * page_size, head_num, head_dim]
```

`k_buffer` 和 `v_buffer` 都是长度为 `layer_num` 的 Python list，不是带 layer
维度的单个 Tensor。第 0 个 page 是 null page，也包含在第一维中。

## 为什么使用扁平 slot

Attention writer 接收扁平位置 `loc`：

```text
slot = page_id * page_size + offset_in_page
```

因此 kernel 可以直接访问：

```python
k_buffer[layer_id][loc]
v_buffer[layer_id][loc]
```

不需要把 `loc` 拆回 `(page_id, offset_in_page)`。普通 MHA field 使用 exact page
stride，`.view(-1, head_num, head_dim)` 只改变 Tensor 元数据，不复制存储。

## 读取接口

- `get_key_buffer(layer_id)`：返回该层 K view；
- `get_value_buffer(layer_id)`：返回该层 V view；
- `get_kv_buffer(layer_id)`：返回 `(K, V)`；
- `_get_key_buffer()` / `_get_value_buffer()`：跳过 layer-wise load 等待的内部版本。

公开 getter 在配置了 `layerwise_load_tracker` 时，会先等待该层从 Host L2 加载
完成。如果底层以 `uint8` 保存 FP8 bit，getter 会先通过 `.view(self.dtype)` 恢复
逻辑 dtype。

## 写入接口

`set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)` 的步骤为：

1. 读取 `layer.layer_id`；
2. 输入 dtype 与 cache dtype 不同时，可先除以 per-tensor scale；
3. 将 K/V 转为逻辑 cache dtype；
4. 必要时重新解释为底层 `store_dtype`；
5. 调用 `store_kv_cache()` 将 K/V scatter 到 `loc`。

写入和 getter 都通过 `_layer_row_view()`，保证异构 head 层使用相同的 row 解释。

## 异构 KV head 数

底层 field 按最大 `head_num` 分配。某层实际 head 数为 `heads_l` 时：

```text
heads_l = max(1, head_num * served_heads / allocation_heads)
```

如果 `heads_l < head_num`，`_layer_row_view()` 将：

```text
[rows, head_num, head_dim]
```

重新解释为：

```text
[rows * head_num / heads_l, heads_l, head_dim]
```

字节数和 page stride 不变，只是 row 数与 head 宽度互换。应区分原始
`self.k_buffer[layer]` 与 backend 通过 `get_key_buffer(layer)` 取得的 view；二者
在异构 head 场景下 shape 可能不同。

## 统计与传输

`get_kv_size_bytes()` 按 `data_ptr()` 去重后统计 K/V 字节，避免 alias field 被重复
计数。传统连续传输 ABI 提供：

- 每层 K/V 的 data pointer；
- 每个 buffer 的总字节数；
- 每个 page 的 item bytes；
- 每层在 K 列表和 V 列表中的 offset。

如果多个 layer view 已经 alias 同一地址，同时又启用旧式 PD per-layer 注册，构造
会失败，因为相同物理字节会被重复注册和传输。通用 plan-based PD 则由基类的
cache contract 描述。

## 关键不变量

- `layer_group_ids` 必须覆盖每层；
- plan 必须包含每层的 K/V field；
- K/V field 必须可展平为连续 token slot；
- writer 和 getter 必须使用相同的 `_layer_row_view()`；
- page 0 是 null page，不计入有效 token 容量。
