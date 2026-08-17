# HybridInklingTokenToKVPool

[子类索引](README.md) · [父类：Hybrid MHA](hybrid-mha.md)

源码：[`kv_cache/hybrid_inkling.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py)

Recipe：[`recipes/inkling.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py)

Inkling pool 在 MHA K/V 之外提供 ShortConv checkpoint accessor。attention groups 由
layer label 生成；recipe 另外声明两个 state checkpoint group：KVConv 与 HiddenConv。

## 字段

每层 attention：

```text
k / v [P, layer_kv_heads_per_rank, head_dim]
```

每层 checkpoint：

```text
kvconv_k / kvconv_v  [sconv_kernel_size - 1, kv_row]
attnconv / mlpconv   [sconv_kernel_size - 1, hidden_size]
```

checkpoint field 使用 `exact_page_stride=False`，可以填入 K/V plane slack。BF16
HiddenConv 较宽时使用独立 hidden plane；FP8 HiddenConv 可 overlay K/V plane。实际
plane、dtype、offset 与 stride都已经写入 plan。

## Accessor

```python
kvconv_checkpoint_buffers(layer_id)
hiddenconv_checkpoint_buffer(layer_id, "attnconv" | "mlpconv")
```

两者把局部 layer 映射到 global continuation id，然后调用 `arena.field(id)` 查表。
field 在 arena 构造时已物化；这些 accessor 不是 lazy binding，也不传 dtype。

`INKLING_FP8_SCONV` 由 recipe 决定 HiddenConv field dtype：未开启为 BF16，开启为
`float8_e5m2`。KVConv 固定 BF16。改变环境变量会改变 plan，不能在 arena 创建后再
切换解释。

## 共享与清零

逐层 KV head 数通过 `CachePoolSpec.layer_kv_head_counts` 交给 MHA row
reinterpretation。checkpoint group 与 attention plane 发生 overlay，因此继承
`requires_page_zeroing=True`。传输必须从 plan 选取 checkpoint field，不能只传 K/V。

## 不变量

- target/draft layer 都由同一个 Inkling recipe 合并规划；
- checkpoint kernel 必须尊重 runtime stride；
- HiddenConv 环境变量在构建 plan 后不可改变其 dtype 语义；
- block reuse 必须清零所有相关 group 字段。
