# CachePool 总览与继承体系

[返回目录](../cache.md) · [子类逐类详解](pools/README.md) · [下一篇：CacheLayout](layout.md)

`CachePool` 定义在
[`python/tokenspeed/runtime/layers/attention/kv_cache/base.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/base.py)，
是 TokenSpeed page-backed cache 的物理存储基类。它根据 `CacheMemoryPlan` 分配
一块共享 `uint8` arena，并把其中不同区域绑定成计算 backend 需要的 Tensor view。

```text
cache recipe
    ↓
CachePoolSpec + CacheMemoryPlan
    ↓
CachePool：共享 arena、field view、runtime contract
    ├── MHA / MLA 计算接口
    ├── sparse index side cache
    ├── recurrent/checkpoint state
    └── model-specific compressed cache
```

## 职责边界

`CachePool` 负责：

- 按 plan 懒分配物理 arena；
- 通过 `field()` 创建 typed、strided、zero-copy Tensor view；
- 发布 scheduler 的 group/page runtime contract；
- 按 group/page 清零；
- 描述 PD 和 Host L2 transfer layout；
- 支持普通 MHA/MLA target 与 draft 共享 backing arena。

具体子类负责：

- field 命名和 runtime dtype；
- backend 看到的 shape；
- K/V、latent、index 或 state 的读写接口；
- 模型特有量化、stride、清零和传输限制。

可以概括为：

```text
CachePool 管“字节在哪里、如何共享”
子类管“kernel 如何把字节看成模型 cache”
```

## 继承树

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

MXFP8 hybrid 类使用多重继承：

- `HybridMHATokenToKVPoolMXFP8` 组合 Hybrid MHA 与 MHA MXFP8；
- `HybridInklingTokenToKVPoolMXFP8` 组合 Inkling checkpoint 与 Hybrid MHA
  MXFP8。

`LayerMappedKVPool` 不在继承树中。它通过组合转发，把调用方 layer id 映射到
内部 pool slot，本身不继承 `CachePool`。

## 子类文档

| 分支 | 类 | 主要扩展 | 详细文档 |
| --- | --- | --- | --- |
| MHA | `MHATokenToKVPool` | 普通按层 K/V | [文档](pools/mha.md) |
| MHA | `MHATokenToKVPoolMXFP8` | FP8 data + UE8M0 scale | [文档](pools/mha-mxfp8.md) |
| MHA sparse | `MSATokenToKVPool` | MiniMax key-only index cache | [文档](pools/msa.md) |
| Hybrid MHA | `HybridMHATokenToKVPool` | MHA history + recurrent state | [文档](pools/hybrid-mha.md) |
| Hybrid MHA | `HybridMHATokenToKVPoolMXFP8` | Hybrid state + MXFP8 | [文档](pools/hybrid-mha-mxfp8.md) |
| Inkling | `HybridInklingTokenToKVPool` | K/V + ShortConv checkpoint | [文档](pools/hybrid-inkling.md) |
| Inkling | `HybridInklingTokenToKVPoolMXFP8` | Inkling + MXFP8 | [文档](pools/hybrid-inkling-mxfp8.md) |
| MLA | `MLATokenToKVPool` | latent KV | [文档](pools/mla.md) |
| MLA sparse | `DSATokenToKVPool` | latent KV + FP8 index-K | [文档](pools/dsa.md) |
| Kimi-K3 | `HybridKDATokenToKVPool` | MLA history + KDA state | [文档](pools/hybrid-kda.md) |
| DeepSeek V4 | `HybridDeepseekV4TokenToKVPool` | SWA/compressed/state/indexer | [文档](pools/hybrid-deepseek-v4.md) |

## Factory 选择关系

具体类由
[`kv_cache/factory.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/factory.py)
根据 config 类型、`CachePoolSpec.family` 和量化开关选择：

| Config / family | 创建的 pool |
| --- | --- |
| `MHAConfig` / `mha` | `MHATokenToKVPool` 或 MXFP8 变体 |
| `MLAConfig` / `mla` | `MLATokenToKVPool` |
| `DSAConfig` / `dsa` | `DSATokenToKVPool` |
| `MSAConfig` / `msa` | `MSATokenToKVPool` |
| `MHAConfig` / `qwen_gdn` | `HybridMHATokenToKVPool` |
| `MHAConfig` / `inkling` | `HybridInklingTokenToKVPool` 或 MXFP8 变体 |
| `MLAConfig` / `kimi_k3` | `HybridKDATokenToKVPool` |
| `deepseek_v4` | `HybridDeepseekV4TokenToKVPool` |

注意：factory 中存在 Qwen GDN MXFP8 的类选择分支，但当前 Qwen cache recipe 会
在准备阶段拒绝 MXFP8，因此实际不能仅凭 factory 分支判断某组合可用。
