# HybridKDATokenToKVPool

[子类索引](README.md) · [父类：MLA](mla.md)

源码：[`kv_cache/hybrid_kda.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_kda.py)

Recipe：[`recipes/kimi_k3.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py)

该类让 Kimi K3 的 MLA history 与 KDA state 成为一个 arena 的不同计算视图。recipe
把 full-attention 层放入 `full_attention`，KDA 层拆成三个
`linear_attention_<n>` state group；draft MLA layer 作为 continuation 加入 full group。

## 字段与接口

MLA layer：

```text
layer.<id>.latent_kv [P, 1, latent_width]
```

KDA layer：

```text
layer.<id>.conv_state       bfloat16
layer.<id>.recurrent_state  float32
```

绑定表合并 MLA latent/scale/rope 与 `conv_state` / `recurrent_state`。plan 未为 KDA
层声明 latent，因此对应 `kv_buffer` 为 `None`；state pair 存入
`_state_buffers_by_layer`。

- MLA 层继续使用 `set_mla_kv_buffer()` / `get_mla_kv_buffer()`；KDA 子类默认开启
  latent write sanitize，防止图捕获 dummy slot 的 NaN 污染；
- KDA 层使用 `get_state_buffers()` 或 `get_component()`；
- `group_id_for_layer()` 返回对应 scheduler group；
- KDA 不支持 `per_token_head` MLA cache。

## Packing 与清零

K3 recipe 将 full-attention packing 固定为 12，并按 MLA plane 字节宽度计算 KDA
packing，使 state group 骑在 MLA plane 中；容量使用 family-specific
`parents_needed()` 与公共二分反推。

该类设置 `requires_page_zeroing=True`，`zero_new_blocks()` 调用
`arena.zero_blocks()`。`get_kv_size_bytes()` 返回整个 arena owner 大小，避免按 overlay
view 重复累加。

## 不变量

- layer type/group id 必须覆盖局部 view；state dtype 来自 plan，不存在额外 map；
- MLA 与 KDA layer 必须走不同 accessor；
- state field 可有 padding，kernel 必须尊重 runtime stride；
- 复用 state block 前必须清零。
