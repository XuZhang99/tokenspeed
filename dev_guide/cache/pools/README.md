# CachePool 子类索引

[返回 Cache 文档目录](../../cache.md) · [查看继承总览](../overview.md)

本目录为每个生产 `CachePool` 子类提供独立说明。`LayerMappedKVPool` 是组合式
layer-id wrapper，不继承 `CachePool`，因此不在此列表。

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

| 类 | 主要用途 | 文档 |
| --- | --- | --- |
| `MHATokenToKVPool` | 普通 MHA K/V cache | [mha.md](mha.md) |
| `MHATokenToKVPoolMXFP8` | MHA block-scaled MXFP8 K/V | [mha-mxfp8.md](mha-mxfp8.md) |
| `MLATokenToKVPool` | MLA latent KV cache | [mla.md](mla.md) |
| `DSATokenToKVPool` | MLA + FP8 稀疏 index-K | [dsa.md](dsa.md) |
| `MSATokenToKVPool` | MHA + MiniMax sparse index-K | [msa.md](msa.md) |
| `HybridMHATokenToKVPool` | MHA history + recurrent state | [hybrid-mha.md](hybrid-mha.md) |
| `HybridMHATokenToKVPoolMXFP8` | Hybrid MHA + MXFP8 | [hybrid-mha-mxfp8.md](hybrid-mha-mxfp8.md) |
| `HybridInklingTokenToKVPool` | Inkling K/V + ShortConv checkpoint | [hybrid-inkling.md](hybrid-inkling.md) |
| `HybridInklingTokenToKVPoolMXFP8` | Inkling + MXFP8 K/V | [hybrid-inkling-mxfp8.md](hybrid-inkling-mxfp8.md) |
| `HybridKDATokenToKVPool` | Kimi-K3 MLA + KDA state | [hybrid-kda.md](hybrid-kda.md) |
| `HybridDeepseekV4TokenToKVPool` | DeepSeek V4 多组压缩 cache | [hybrid-deepseek-v4.md](hybrid-deepseek-v4.md) |

Factory 的主要选择关系：

| Config / family | Pool |
| --- | --- |
| `MHAConfig` / `mha` | `MHATokenToKVPool` 或 MXFP8 变体 |
| `MLAConfig` / `mla` | `MLATokenToKVPool` |
| `DSAConfig` / `dsa` | `DSATokenToKVPool` |
| `MSAConfig` / `msa` | `MSATokenToKVPool` |
| `MHAConfig` / `qwen_gdn` | `HybridMHATokenToKVPool` |
| `MHAConfig` / `inkling` | `HybridInklingTokenToKVPool` 或 MXFP8 变体 |
| `MLAConfig` / `kimi_k3` | `HybridKDATokenToKVPool` |
| `deepseek_v4` | `HybridDeepseekV4TokenToKVPool` |
