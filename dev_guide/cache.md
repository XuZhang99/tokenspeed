# CachePool 文档

本组文档说明 TokenSpeed page-backed cache 从模型 recipe、容量无关布局、容量绑定
计划，到运行时 Tensor view 和 scheduler/transfer 集成的完整链路。

```text
CacheFieldSpec
    ↓ solve_cache_layout()
CacheLayout
    ↓ with_num_lcm_blocks()
CacheMemoryPlan
    ↓ 与逐层和 scheduler 元数据组成 CachePoolSpec
CachePoolSpec
    ↓ create_cache_pool() / CachePool.field()
MHA / MLA / hybrid Tensor views
    ↓ runtime_contract
Scheduler / PD / Host L2
```

## 文档目录

| 文档 | 内容 |
| --- | --- |
| [CachePool 总览与继承体系](cache/overview.md) | `CachePool` 职责、继承树、factory 选择关系和逐类文档入口。 |
| [CachePoolSpec](cache/pool-spec.md) | Recipe 与 pool factory 的交接对象、字段语义、创建路径、容量概念和 target/draft 视图。 |
| [CachePool 子类逐类详解](cache/pools/README.md) | 11 个生产子类的独立文档，覆盖字段、shape、读写、量化、传输和限制。 |
| [CacheLayout](cache/layout.md) | `CacheFieldSpec`、容量无关的单父块几何、`solve_cache_layout()` 和 packing 求解。 |
| [CacheMemoryPlan](cache/memory-plan.md) | 容量绑定后的 groups、planes、fields、overlay 关系和 arena 容量。 |
| [Arena 与 `field()` 绑定](cache/field-binding.md) | arena 懒分配、typed/strided Tensor view、null page offset 和子类绑定示例。 |
| [CachePool 运行时集成](cache/runtime.md) | 构造参数、scheduler runtime contract、backing pool、清零、PD/Host L2 和生命周期。 |

建议按表格顺序阅读。只想理解 MHA K Cache shape 时，可以直接阅读
[MHATokenToKVPool](cache/pools/mha.md)；只想理解字节布局如何变成 Tensor 时，可以
直接阅读 [Arena 与 `field()` 绑定](cache/field-binding.md)。

主要源码入口：

- [`kv_cache/base.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/base.py)
- [`kv_cache/recipes/setup.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/setup.py)
- [`kv_cache/recipes/plan.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)
- [`kv_cache/factory.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/factory.py)
