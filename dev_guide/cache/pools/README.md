# CachePool 子类索引

[返回 Cache 文档目录](../../cache.md) · [查看对象边界](../overview.md)

所有类都是一个 `CacheArena` 上的计算 layer window；它们不分配 arena、不发布
runtime contract，也不决定 field dtype。每个类只声明 `layer_plane_bindings` 并提供
kernel-facing 访问接口。

```text
CachePool
├── MHATokenToKVPool
│   ├── MHATokenToKVPoolMXFP8
│   ├── MSATokenToKVPool
│   └── HybridMHATokenToKVPool
│       ├── HybridMHATokenToKVPoolMXFP8
│       └── HybridInklingTokenToKVPool
│           └── HybridInklingTokenToKVPoolMXFP8
├── MLATokenToKVPool
│   ├── DSATokenToKVPool
│   └── HybridKDATokenToKVPool
└── HybridDeepseekV4TokenToKVPool
```

| 类 | 计算视图 | 文档 |
| --- | --- | --- |
| `MHATokenToKVPool` | 普通 MHA K/V | [mha.md](mha.md) |
| `MHATokenToKVPoolMXFP8` | FP8 data + UE8M0 scale | [mha-mxfp8.md](mha-mxfp8.md) |
| `MLATokenToKVPool` | MLA latent/nope/rope | [mla.md](mla.md) |
| `DSATokenToKVPool` | MLA + block-split sparse index-K | [dsa.md](dsa.md) |
| `MSATokenToKVPool` | MHA + MiniMax sparse index-K | [msa.md](msa.md) |
| `HybridMHATokenToKVPool` | MHA history + recurrent state | [hybrid-mha.md](hybrid-mha.md) |
| `HybridMHATokenToKVPoolMXFP8` | hybrid state + MXFP8 | [hybrid-mha-mxfp8.md](hybrid-mha-mxfp8.md) |
| `HybridInklingTokenToKVPool` | MHA + ShortConv checkpoint | [hybrid-inkling.md](hybrid-inkling.md) |
| `HybridInklingTokenToKVPoolMXFP8` | Inkling + MXFP8 | [hybrid-inkling-mxfp8.md](hybrid-inkling-mxfp8.md) |
| `HybridKDATokenToKVPool` | MLA history + KDA state | [hybrid-kda.md](hybrid-kda.md) |
| `HybridDeepseekV4TokenToKVPool` | SWA/compressed/state/indexer | [hybrid-deepseek-v4.md](hybrid-deepseek-v4.md) |

创建关系以 [`kv_cache/factory.py`](../../../python/tokenspeed/runtime/layers/attention/kv_cache/factory.py)
为准。新增 pool 时通常需要同时更新 recipe family、factory、backend consumer family、
单测和本索引。
