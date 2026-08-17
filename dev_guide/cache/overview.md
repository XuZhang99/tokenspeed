# Cache 总览：Recipe、Arena 与 Pool

[返回目录](../cache.md) · [CachePoolSpec](pool-spec.md) · [下一篇：CacheLayout](layout.md)

当前实现把「几何与所有权」和「kernel 计算视图」彻底分开：

```text
CacheRecipe.setup()
  └─ CachePoolSpec.memory_plan
          └─ CacheArena            唯一的物理所有者
              ├─ buffer
              ├─ plan / field views
              └─ CacheRuntimeContract
                   ▲
                   ├─ target CachePool（layer window）
                   └─ draft  CachePool（continuation layer window）
```

## `CacheRecipe`：统一的构建流水线

[`recipes/base.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/base.py)
中的 `CacheRecipe.setup()` 是唯一声明流水线顺序的地方：

1. `layer_types` / `group_ids` 描述 target 层以及紧随其后的 draft continuation 层；
2. `groups()` 生成 `(CacheGroupSpec, CacheFieldSpec[])`；
3. `pack()` 求一个 LCM parent 的容量无关布局；
4. family 选择 `num_lcm_blocks` 和 `token_capacity`；
5. `CacheLayout.bind()` 生成 `CacheMemoryPlan`；
6. 返回 `CacheSetup(CachePoolSpec, ...)`。

各 family 只覆盖统一的 seam，例如 `fields_for_layer()`、`packing()`、
`check_layout()`、`parents_needed()` 和 `workspace_bytes()`，不能复制一条私有 setup
流水线。family 到 recipe 的唯一注册表是 `recipes/setup.py::_RECIPES`。

## `CacheArena`：唯一的物理所有者

[`kv_cache/arena.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)
中的 `CacheArena` 负责：

- 按 `plan.arena_bytes` 一次性分配连续的 `uint8` buffer；
- 构造时为 `plan.fields` 中的每个 field 建立 typed/strided Tensor view；
- 保存 `CacheMemoryPlan`，提供 `field()` 与字节偏移查询；
- 将 group 语义与 plan 的 page count/packing 组合成唯一的
  `CacheRuntimeContract`；
- 按 group/block 清零，或在 sleep/wake 后清空整个 arena；
- 向 PD 暴露可传输的 owner buffer。

arena 不再延迟分配或延迟绑定字段；field dtype 也不由 pool 在运行时重新提供。

## `CachePool`：计算视图

[`kv_cache/base.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/base.py)
中的 `CachePool` 不拥有 buffer、plan 或 runtime contract。它保存的是真正属于计算
视图的信息：

- kernel 读取缓存时使用的逻辑 dtype；
- 当前视图的 layer 数与 `field_layer_offset`；
- 每层 K/V、latent、scale 或 state buffer 列表；
- kernel-facing getter、setter 和 layer-wise transfer hook。

子类用 `layer_plane_bindings` 声明「plane 名 → buffer 属性名」。基类
`_bind_layer_planes()` 遍历 plan，把当前 layer window 内已经由 arena 物化的字段装进
对应 list；plan 未声明的 plane 保持 `None`。这使「某层有哪些字段」只由 plan 决定。

## 继承体系

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

逐类接口见 [Pool 子类索引](pools/README.md)。

## Factory 分工

[`kv_cache/factory.py`](../../python/tokenspeed/runtime/layers/attention/kv_cache/factory.py)
有两个独立入口：

- `create_cache_arena(spec, ...)`：对合并 `CachePoolSpec` 只调用一次；
- `create_cache_pool(spec, config, arena, ...)`：为 target/draft 分别创建 layer view。

| Config / family | Pool |
| --- | --- |
| `MHAConfig` / `mha` | `MHATokenToKVPool`，可选 MXFP8 |
| `MHAConfig` / `qwen_gdn` | `HybridMHATokenToKVPool`，可选 MXFP8 类 |
| `MHAConfig` / `inkling` | `HybridInklingTokenToKVPool`，可选 MXFP8 |
| `MLAConfig` / `mla` | `MLATokenToKVPool` |
| `MLAConfig` / `kimi_k3` | `HybridKDATokenToKVPool` |
| `DSAConfig` / `dsa` | `DSATokenToKVPool` |
| `MSAConfig` / `msa` | `MSATokenToKVPool` |
| `deepseek_v4` | `HybridDeepseekV4TokenToKVPool` |

`registry.create_attn_components()` 先准备合并 spec、创建 arena，再把 target 和 draft
的局部 layer id 映射到同一个 plan 的全局 continuation layer。

## 维护时的边界

- 物理几何只从 `arena.plan` 读取，不能在 pool 上维护镜像；
- scheduler contract 只由 arena 发布，pool view 不得重新发布；
- dtype 写在 `CacheFieldSpec` / `CacheFieldLayout`，`arena.field()` 没有 dtype 参数；
- group id 在 `(CacheGroupSpec, fields)` 声明中只写一次，packing 只由 layout 决定；
- target/draft 共享必须通过同一个 `CacheArena`，不再使用 `backing_pool` 模式。
