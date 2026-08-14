# MSATokenToKVPool

[子类索引](README.md) · [父类：MHATokenToKVPool](mha.md)

源码：[`kv_cache/msa.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/msa.py)

## 定位与创建条件

`MSATokenToKVPool` 用于 MiniMax Sparse Attention，在普通 MHA K/V 之外增加
key-only 稀疏索引 side cache：

```text
CachePool
└── MHATokenToKVPool
    └── MSATokenToKVPool
```

Factory 在 `config` 是 `MSAConfig` 时创建，传入 `index_head_dim`、模型
`index_dtype` 和 `sparse_layer_ids`。

## 继承的 MHA 部分

父类仍为每层绑定：

```text
layer.<id>.k
layer.<id>.v
```

并提供相同的扁平 slot shape、getter、`set_kv_buffer()` 和 layer-wise load 同步。
MSA 没有改变普通 K/V 的数据布局。

## Index-K side cache

只有 `indexed_layer_ids` 中的层绑定：

```text
layer.<id>.index_k
```

运行时保存在 dict：

```python
index_k_buffer: dict[int, Tensor]
```

每个 Tensor 被解释为：

```text
[num_slots, index_head_dim]
dtype = index_dtype
```

它与 DSA 的 FP8 block-split index-K 不同：MSA 类本身不执行专用量化，也不把
value 与 scale 打包进 `uint8` row；具体 dtype 直接来自 `MSAConfig` 的模型 dtype。

## 初始化顺序

构造函数先保存 sparse layer 集合并建立空 dict，再调用 MHA 父类完成 K/V arena
绑定，最后在同一个 memory-saver region 语义下绑定 index-K field。Recipe 必须只为
配置的 sparse layer 规划对应 field。

## 访问接口

```python
get_index_k_buffer(layer_id)
```

会：

1. 等待可选的 `layerwise_load_tracker`；
2. 检查 layer 是否存在于 dict；
3. 返回该层 index-K view。

访问非 sparse layer 会抛出 `RuntimeError`，而不是返回 `None`，防止 backend 静默
跳过错误的索引配置。

## 显存统计

父类把 K/V 分开统计。MSA 将所有 index-K 字节计入 key 一侧：

```text
reported_key_bytes = K bytes + index-K bytes
reported_value_bytes = V bytes
```

这只是统计归类，不表示 index-K 与普通 K field 是同一个逻辑 Tensor。

## 当前传输限制

`get_contiguous_buf_infos()` 明确抛出 `NotImplementedError`：传统 MiniMax sparse
cache transfer ABI 尚未定义如何携带 index-key side cache。不能把父类只覆盖 K/V
的连续传输描述误用于 MSA。

Plan-based `CachePool` arena、scheduler runtime contract 和通用字段绑定仍然有效；
但需要传统连续 buffer 元数据的调用路径必须先补齐 side-cache contract。

## 关键不变量

- `indexed_layer_ids` 必须与 recipe 中的 `index_k` fields 一致；
- 非 sparse layer 不应调用 `get_index_k_buffer()`；
- index-K shape 的最后一维必须等于 `index_head_dim`；
- 不要把 DSA 的 block-split/FP8 假设套用到 MSA；
- 使用 Host/PD 传输前必须确认所走 ABI 是否覆盖 index-K。
