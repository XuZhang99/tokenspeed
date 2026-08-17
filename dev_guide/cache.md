# Cache 开发文档

本组文档对应当前统一缓存流水线：所有模型族都执行同一条
`layers → group → pack → bind` 路径；一个合并模型只创建一个 `CacheArena`，target
和 draft 的 `CachePool` 都只是这个 arena 上的计算视图。

```text
layer_types + group_ids
        │
        ▼ group()
(CacheGroupSpec, CacheFieldSpec[])[]
        │
        ▼ pack()
CacheLayout                         # 单个 LCM parent，容量无关
        │
        ▼ bind(num_lcm_blocks)
CacheMemoryPlan                     # 容量绑定、dtype 完整
        │
        ├── CacheArena              # 唯一分配者、字段 view、runtime contract
        └── CachePoolSpec
                 │
                 ▼ create_cache_pool()
        target / draft CachePool    # layer window，不拥有内存
```

## 文档目录

| 文档 | 内容 |
| --- | --- |
| [总览与对象边界](cache/overview.md) | `CacheRecipe`、`CacheArena`、`CachePool` 的职责及 factory 关系。 |
| [CachePoolSpec](cache/pool-spec.md) | Recipe 向 arena/pool factory 交付的不可变描述。 |
| [CacheLayout](cache/layout.md) | `CacheFieldSpec`、`CacheGroupSpec`、`group()`、`pack()` 与 packing。 |
| [CacheMemoryPlan](cache/memory-plan.md) | `bind()` 后的 group、plane、field、dtype、null block 与字节偏移。 |
| [Arena 与字段绑定](cache/field-binding.md) | eager allocation、`as_strided` view、token-row 折叠和 layer-plane 绑定。 |
| [运行时集成](cache/runtime.md) | scheduler contract、target/draft 共享、清零、PD 与 Host L2。 |
| [Pool 子类](cache/pools/README.md) | MHA、MLA、DSA、MSA、hybrid、Inkling、KDA 和 DeepSeek V4。 |
| [Qwen3.5 LCM Cache](qwen35-lcm-cache.md) | full attention + GDN 的分组、packing、调度、state 双索引和 MTP。 |

术语约定见 [KV Cache 管理机制](kvcache-management.md)，LCM 数学与扩展 recipe 的
方法见 [LCM 两级分配](lcm.md)。英文的逻辑/物理分层规范以
[`docs/design/cache-concepts.md`](../docs/design/cache-concepts.md) 为准。

主要源码入口：

- [`recipes/base.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/base.py)
- [`recipes/spec.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py)
- [`recipes/plan.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)
- [`arena.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)
- [`base.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/base.py)
- [`factory.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/factory.py)
