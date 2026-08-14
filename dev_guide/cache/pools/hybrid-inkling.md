# HybridInklingTokenToKVPool

[子类索引](README.md) · [父类：HybridMHATokenToKVPool](hybrid-mha.md)

源码：[`kv_cache/hybrid_inkling.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py)

Recipe：[`recipes/inkling.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py)

## 定位与创建条件

该类为 Inkling 模型在 MHA K/V 之外暴露 ShortConv checkpoint view：

```text
CachePool
└── MHATokenToKVPool
    └── HybridMHATokenToKVPool
        └── HybridInklingTokenToKVPool
```

Factory 在 `MHAConfig + spec.family == "inkling"` 且未启用 MXFP8 时选择它。

虽然继承自 Hybrid MHA，Inkling recipe 为每层都规划 attention K/V；父类在这里
主要提供统一 arena、group 映射、异构 KV head view、page 清零和 contract 能力。
ShortConv checkpoint 使用独立 group，但可与 K/V plane overlay。

## K/V 字段

每层首先具有普通 MHA 字段：

```text
layer.<id>.k
layer.<id>.v
```

Field 页内 shape 使用该层实际 KV head 数：

```text
[logical_block_tokens, layer_kv_heads, head_dim]
```

Recipe 将逐层 head 数传入 `layer_kv_head_counts`。父类底层按统一分配宽度管理，
`_layer_row_view()` 在读取和写入时重新解释窄 head 层。

## ShortConv checkpoint 字段

每层额外规划四个 field：

```text
layer.<id>.kvconv_k
layer.<id>.kvconv_v
layer.<id>.attnconv
layer.<id>.mlpconv
```

逻辑用途：

| Field | 内容 | 页内 shape |
| --- | --- | --- |
| `kvconv_k` | K 侧 ShortConv checkpoint | `[checkpoint_rows, kv_heads * head_dim]` |
| `kvconv_v` | V 侧 ShortConv checkpoint | `[checkpoint_rows, kv_heads * head_dim]` |
| `attnconv` | attention hidden-state checkpoint | `[checkpoint_rows, hidden_size]` |
| `mlpconv` | MLP hidden-state checkpoint | `[checkpoint_rows, hidden_size]` |

`checkpoint_rows = sconv_kernel_size - 1`。这些 field 都设置
`exact_page_stride=False`，kernel 必须使用 runtime stride，从而允许较小 checkpoint
利用 K/V plane 中的 slack。

## Plane overlay

Recipe 默认让：

```text
kvconv_k  ↔ K plane
kvconv_v  ↔ V plane
```

当 hidden-conv 使用 1-byte FP8 时，`attnconv/mlpconv` 也可以分别 overlay K/V
plane；使用 2-byte BF16 时则使用独立的 `hidden_k/hidden_v` plane。最终选择由
`hiddenconv_element_size` 决定，实际 offset/stride 仍由 `CacheLayout` 求解。

## Checkpoint accessor

```python
kvconv_checkpoint_buffers(layer_id) -> (kvconv_k, kvconv_v)
```

按需调用 `field()` 绑定 BF16 checkpoint view。

```python
hiddenconv_checkpoint_buffer(layer_id, component)
```

其中 `component` 只能是：

```text
attnconv
mlpconv
```

其他值会抛出 `ValueError`。这些 accessor 采用延迟 field 绑定，但底层仍复用同一
arena 和 `_fields` 注册表。

## HiddenConv dtype

环境变量控制 hidden-conv checkpoint dtype：

```text
INKLING_FP8_SCONV=0 或未设置 → torch.bfloat16
INKLING_FP8_SCONV!=0         → torch.float8_e5m2
```

KVConv checkpoint 固定使用 BF16。Recipe 使用相同环境变量选择 element size，pool
使用它选择 runtime dtype；两边必须一致，否则 `field()` 的 itemsize 校验会失败。

## Page 清零与传输

Checkpoint group 与 K/V 发生 plane overlay，因此继承：

```python
paged_cache_requires_page_zeroing = True
```

Scheduler 分配的新 page 必须在使用前按 group 清零。传输应走 plan-based cache
contract，不能只注册普通 MHA K/V 而遗漏 checkpoint fields。

## 关键不变量

- recipe 的逐层 KV head 数必须与 pool 的 head reinterpretation 一致；
- ShortConv field 使用 runtime stride，不能假定 payload-sized page stride；
- `INKLING_FP8_SCONV` 必须在 recipe 和 runtime 构造期间保持一致；
- checkpoint accessor 只接受定义好的 component；
- overlay page 在复用前必须清零。
