# LCM 两级 KV-Cache 分配机制

[Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://dl.acm.org/doi/10.1145/3731569.3764823)

##### commit-id: 638b9de0e698446b5e50a2a9508778b2f59473f6

本文详细讲解 TokenSpeed 中 **LCM（two-level）KV-cache 分配体系**——一块共享物理
arena + 两级打包的 KV 内存布局与分配机制。它最初由 PR #804（`jenga two level
allocation`）引入，后经 PR #949（`Refactor/kv cache spec single source`）等重构，
从早期散落的 `configs/lcm_*.py` + `kv_cache/lcm*.py` 文件统一迁入
`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/` 这个 **cache recipe**
包，术语也从 `Lcm*` 前缀改成中性的 `CacheLayout` / `CacheMemoryPlan` / `CachePool`。
**LCM 算法本身逐字未变**（`_solve_packing` / `_check_exact_page_strides` /
`exact_page_stride` / parent LCM 对齐循环都还在），而且现在是**唯一**的 KV-cache
分配路径——不再有 radix / flat-slab 旧体系可选。本文可独立阅读，是
`kvcache-management.md` 第 4 节的完整展开。

---

## 1. LCM 是什么，解决什么问题

### 1.1 背景：异构 cache 的对齐难题

传统 attention 模型每个 token 只有一种 KV：K/V 两块张量，按 page 分配即可。但
hybrid 模型一层可能是 **full-attention**（存 history K/V 或 MLA latent），另一层是
**linear-attention**（存 recurrent state + conv state），甚至还叠加 draft 模型的
history、Inkling 的 ShortConv checkpoint 等。这些 cache：

- **每 token 的字节数天差地别**：一段 recurrent state 是「每个 sequence 一份、和
  序列长度无关」的定长快照，而 history K/V 是「每 token 一行」线性增长。
- **kernel 对 stride 的假设不同**：有的 kernel 按隐式 payload-sized stride 走页
  （必须精确），有的 kernel 在 launch 时读取运行时 `stride(0)`（可吸收 slack）。

如果给每种 cache 单独开一块 pool，会碰到两个问题：(1) 内存被切成多块，碎片化、
利用率低；(2) 调度器要同时管理多套 page 表、多套容量约束，复杂且易错。

### 1.2 LCM 的方案：一块 arena + 两级打包

LCM 把所有这些异构 cache 打包进 **一块 budget 大小的连续物理 arena**，用单一
`cache_budget_bytes` 精确推导整个几何。核心思想是**两级（two-level）**：

- **上层单位 = parent LCM block**：arena 被切成 `num_lcm_blocks` 个等大的 parent
  block，parent 是调度器**唯一**的分配/驱逐/调度单位，固定容纳
  `logical_block_tokens` 个逻辑 token。该值现在是**每个 recipe 自选的参数**（不再是
  全局常量）：GDN / Inkling 用 128（`recipes/qwen35.py:132`、`recipes/inkling.py:237`），
  Kimi-K3 用 `_KIMI_K3_LOGICAL_BLOCK_TOKENS`（`recipes/kimi_k3.py`），DeepSeek-V4 用
  256（`recipes/deepseek_v4.py:24`）。
- **下层单位 = per-group child page**：每个 cache group 在一个 parent 里按各自的
  **packing 因子 `cache_blocks_per_lcm_block`** 容纳固定整数个子 page。

分配**一个 parent** 就同时为**所有** group（history KV、recurrent/conv state、
draft history…）预留好各自数量的 page——调度器只需管理 parent 一个维度的整数，
每个 group 的物理页号由 parent id 乘以该 group 的 packing 因子直接算出。

### 1.3 名字由来："LCM" = 最小公倍数对齐

parent block 的字节大小必须能被**每个 group 的 packing 因子整除**（这样才能把
parent 均匀切成整数个 child page）。规划器通过对所有 packing 因子求
**最小公倍数（Least Common Multiple）**来定 parent 的对齐
（`recipes/plan.py:637`-`640`，在 `solve_cache_layout` 内）：

```python
parent_alignment = alignment
for count in packing.values():
    parent_alignment = parent_alignment // math.gcd(parent_alignment, count) * count
lcm_block_bytes = _align_up(sum(plane_bytes.values()), parent_alignment)
```

即 `parent_alignment = lcm(alignment, *packing.values())`。这个 LCM 对齐正是整个
体系名字的来源。hybrid recipe 传入的 `alignment=256`（`recipes/ordinary.py:252`）。

> **旧体系已删除**：flat-slab 时代（`flat_hybrid.py` / `flat_state_slabs.py` /
> `hybrid_cache_plan.py` / `flat_memory_plan.py`）以及 radix / flat 双后端都已被
> 无条件的两级 cache 取代。**当前 LCM 是唯一路径**：没有匹配 recipe 的模型会在
> `create_attn_components` 直接 raise（`registry.py:860`-`864`）。

---

## 2. 核心概念与术语

| 术语 | 含义 |
|---|---|
| **parent LCM block** | arena 的分配/调度单位，固定 `logical_block_tokens` 逻辑 token。共 `num_lcm_blocks` 个（不含 null）。 |
| **cache group** | 一组共享同一 retention/packing 的层的集合，如 `full_attention` / `linear_attention` / `kvconv` / `hiddenconv`。 |
| **packing 因子** | `cache_blocks_per_lcm_block`，一个 parent 里某 group 塞几个 child page。 |
| **child page / cache block** | group 的最小物理页；一个 parent 里某 group 有 `packing` 个。 |
| **plane** | arena 内的一段连续区域，同一 plane 被多个 group 的同名 field **overlay**（错位复用）共享。plane-major 布局。 |
| **field** | 一个具体张量视图的几何描述（layer 的 k / v / latent_kv / conv_state 等），归属某 group + 某 plane。 |
| **null LCM block（parent 0）** | 永不调度的哨兵 parent，backing 逻辑 null page 0。`arena_bytes` 含它，`num_lcm_blocks` 不含。 |
| **logical_block_tokens** | 逻辑页 token 数，per-recipe 参数（GDN/Inkling 128、DSV4 256）。 |

**两级映射直觉**：某 group（packing = K）的 child page 号 = `parent_id * K + slot`
（slot ∈ [0, K)）；每组 `page_count = 1 + num_lcm_blocks * K`（`recipes/plan.py:251`
一带的 `with_num_lcm_blocks`，`1 +` 是 null parent 的哨兵页）。

---

## 3. 几何层 `recipes/plan.py`

`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py`——**纯整数几何，
不 import torch**（模块 docstring：*"Pure integer geometry for one shared LCM cache
arena."*）。它只负责回答「arena 有多大、每个 field 落在哪个字节偏移、page stride
多少」。这是整个体系的核心，所有 LCM/packing 数学都集中在这一个文件里。

### 3.1 数据类

- `CacheGroupLayout`（`:38`）：per-group 结果——`group_id` +
  `cache_blocks_per_lcm_block`（packing）+ `page_count`。
- `CachePlaneLayout`（`:45`）：`plane_id` + `bytes_per_lcm_block` + `arena_offset_bytes`。
- `CacheFieldLayout`（`:52`）：**输出**——solved 后的 field 放置：`plane_id` /
  `shape` / `element_size` / `field_offset_bytes` / `page_stride_bytes`
  （`.payload_bytes` 在 `:62`）。
- `CacheMemoryPlan`（`:67`）：**容量绑定**的最终产物（旧 `LcmMemoryPlan`），含
  `logical_block_tokens` / `lcm_block_bytes` / `num_lcm_blocks` / `groups` / `planes`
  / `fields`；`arena_bytes`（`:81`，`= (num_lcm_blocks + 1) * lcm_block_bytes`，`+1` 是
  null parent）、`group()`/`field()`/`plane()` 查询（`:84`-`100`）、`capacity_report()`
  （`:102`，per-group dead-bytes / binding-utilization 诊断）。
- `CacheFieldSpec`（`:198`）：**输入**配方——`group_id` / `field_id` / `plane_id` /
  `shape` / `element_size`，外加 `exact_page_stride: bool = True`（`:206`）和
  `page_stride_alignment_bytes: int = 1`（`:210`）。`.payload_bytes` 在 `:213`。
- `CacheLayout`（`:218`）：**容量无关**的单 parent 字节几何（`lcm_block_bytes` /
  `group_packing` / `plane_bytes` / `fields`）。`with_num_lcm_blocks(n)`（`:227`）把它
  变成一个 `CacheMemoryPlan`（每组 `page_count = 1 + n*count`，plane
  `arena_offset_bytes` 累加）。**Layout↔Plan 的这个拆分是新的**：几何先算一次（与容量
  无关），再乘上 `num_lcm_blocks` 得到容量绑定的 plan。

### 3.2 `exact_page_stride`：两类 kernel

`CacheFieldSpec.exact_page_stride`（`:206`）区分 field 的 kernel 对 stride 的假设：

- `True`（默认）：kernel 按**隐式 payload-sized stride** 走页——page stride 必须
  精确等于 `payload_bytes`，否则读错行。history K/V、MLA latent、scale field 属此类。
- `False`：kernel 在 launch 时读取运行时 `stride(0)`，可以吸收其它 field 在 plane
  里留下的 slack。conv/ssm state、Inkling 的 kvconv/hiddenconv 属此类
  （`recipes/inkling.py:89`/`:97`/`:105` 均设 `exact_page_stride=False`）。

### 3.3 求解入口 `solve_cache_layout(...)`（`:457`）

旧 `plan_lcm_fields` 的直接后继。签名：`solve_cache_layout(fields, *,
logical_block_tokens, cache_blocks_per_lcm_block=None, alignment=1,
max_padding_fraction=0.25) -> CacheLayout`。规划一个 **plane-major** arena，field 在
cache group 之间 overlay。关键行为：

1. **packing 推导**：
   - `_solve_packing`（`:363`）：对**同一 plane 上 exact 字节**建立 group 间比例
     约束（并查集 + 分数化简），推出让各 group page stride 精确一致的整数 packing。
   - `_packing_by_group_ratio`（`:425`）：fallback，按 `largest_payload //
     group_payload` 给比例。
   - 显式 `cache_blocks_per_lcm_block` 覆盖以上（供 draft 复用 target 几何，或
     `_ordinary_setup` 强制 packing=1）。
2. **plane / parent 字节**：`plane_bytes` 逐 plane 对齐；`lcm_block_bytes` 对所有
   packing 求 LCM 对齐（`:637`-`640`，见 §1.3）。
3. **校验（fail loud）**：
   - `_check_exact_page_strides`（`:433`，调用点 `:635`）：exact field 的
     `plane_bytes // packing` 必须 == `payload_bytes`，否则报错并提示是哪个 field 把
     plane 撑大了。
   - padding fraction ≤ `max_padding_fraction`（`:654`-`659`）。
   - kernel page id 不超 `_MAX_KERNEL_PAGE_ID = 2^31 - 1`（`:34`，检查在
     `with_num_lcm_blocks` 的 `:237`）；`lcm_block_bytes` 不超
     `_MAX_LCM_BLOCK_BYTES = 2^63 - 1`（`:33`）。
4. **draft 合并**：`merge_continuation_layers`（`:272`）把 draft 模型的 per-layer 向量
   接在 target 之后（draft = 「一个大模型的后续层」）；`continue_layer_fields`（`:326`）
   把 draft `layer.i` 用正则重编号成 `layer.{first+i}`。

`_align_up(value, alignment)` 在 `:359`。

---

## 4. 布局 recipe（各 model family）

每个 model family 有一个 recipe 文件，把层结构翻译成一串 `CacheFieldSpec`，喂给
`solve_cache_layout`。recipe 只**选 packing 策略**（exact-ratio / power-of-two /
pinned）并造 field，真正的 LCM 求解都委托给 `plan.py`。

### 4.1 `recipes/ordinary.py`——MHA / MLA / DSA / MSA（最大的一次旧→新合并）

- `mla_cache_fields`（`:15`）/ `mha_cache_fields`（`:42`，产 `layer.{i}.k/v` field +
  mxfp8 scale field）/ `draft_cache_fields`（`:117`）。
- `build_hybrid_cache_setup`（`:216`）：LCM sizing 驱动——调
  `solve_cache_layout(alignment=256)`（`:252`），`num_lcm_blocks = usable_cache_bytes
  // lcm_block_bytes - 1`（`:261`，`-1` 是 null parent），再套 token 上限，返回
  `CacheSetup`。被 inkling + qwen35 复用。
- `_ordinary_setup`（`:343`）：profiled-bytes 路径，强制 packing=1（`cache_blocks_per
  _lcm_block={gid:1}`、`alignment=1`），供 mha/mla/dsa/msa 单一 group 使用。
- `prepare_ordinary_cache(*, family, ...)`（`:510`）：旧 `_prepare_mha` 的后继，
  按 config 类型分派 `_mha_fields`（`:471`）/ `_mla_fields`（`:552`）等。

### 4.2 `recipes/kimi_k3.py`——Kimi-K3（MLA + KDA）

- `solve_kimi_k3_cache_layout`（`:235`）：把 `FULL_ATTENTION` packing 钉死为 12
  （`_KIMI_K3_MLA_PACKING = 12`，`:29`），从 MLA plane 字节推导 KDA `linear_packing`
  （`:264`-`278`），调 `solve_cache_layout(alignment=256)`，最后 assert 结果 packing
  等于钉死值（`:288`）。draft MLA 层经 `continue_layer_fields` 接入。
- `kimi_k3_token_capacity_for_cache_pool`（`:344`）：二分反推 token 容量。
- `prepare_kimi_k3_cache`（`:388`）：旧 `_prepare_kimi_k3` 的后继。

### 4.3 `recipes/deepseek_v4.py`——DeepSeek-V4

- `solve_deepseek_v4_memory_layout`（`:77`）：用 **2 的幂** packing 而非 exact LCM
  （`:85`-`86` 注释解释 exact 比例会因巨大 LCM「撑爆 parent」），仍调
  `solve_cache_layout(alignment=256)`。
- `prepare_deepseek_v4_cache`（`:199`）。byte-formula 与 spec builder 在
  `recipes/deepseek_v4_cache_spec.py`（`DeepseekV4CacheLayout`:155、
  `deepseek_v4_cache_layout_from_config`:210、`build_v4_cache_specs`:318）。

### 4.4 `recipes/inkling.py` / `recipes/qwen35.py`——hybrid state 家族

- `inkling.py`：`inkling_cache_fields`（`:14`，KV plane + `kvconv`/`hiddenconv` state
  field，均 `exact_page_stride=False`），`prepare_inkling_cache`（`:225`）委托
  `build_hybrid_cache_setup`。
- `qwen35.py`：`qwen_gdn_cache_fields`（`:41`，full-attn history + GDN recurrent
  state），`prepare_qwen35_cache`（`:109`）委托 `build_hybrid_cache_setup`。

---

## 5. 调度器接口 `recipes/spec.py` + `recipes/cache_runtime.py`

`solve_cache_layout` 产出的是 arena 字节几何；scheduler 需要的是**每组多少页**。这两
个文件把几何翻译成调度器契约。

### 5.1 `recipes/spec.py`——group spec 与页数

- `PagedCacheGroupSpec`（`:29`）：调度器契约单位——`group_id` / `retention`
  （`full_history` / `sliding_window`）/ `rows_per_page` / `entry_stride_tokens` /
  `sliding_window_tokens` / `family`（`history` / `state`）/
  `cache_blocks_per_lcm_block`（`:38`，surface 给调度器的 packing 因子）/
  `transfer_policy`。
- `validate_scheduler_config`（`:64`）：每个 pool 必须从 recipe 发布一个
  `runtime_contract`，否则 raise。
- `compute_paged_cache_group_page_counts`（`:105`）：per-group child-page 数
  （history / state / sliding 各自公式）。
- `build_paged_cache_group_specs`（`:593`）、`group_specs_from_layer_types`（`:474`）、
  `apply_pd_transfer_policies`（`:572`）等辅助。

### 5.2 `recipes/cache_runtime.py`——发布出去的运行时契约

- `PagedCacheRuntimeContract`（`:57`）：`block_size` / `num_lcm_blocks` /
  `token_capacity` / `group_specs` / `group_page_counts`。`__post_init__`（`:64`）强制
  `group_page_counts == num_lcm_blocks * cache_blocks_per_lcm_block + 1`（per group）——
  这正是把调度器的 page id 绑到 LCM plan 的不变式。

---

## 6. Setup 编排 `recipes/setup.py`

`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/setup.py`——从
`cache_budget_bytes` 同时给 target + draft arena 定尺寸的结果容器与 family 分派。

### 6.1 产物数据类

- `CacheModelFamily`（`:53`）：`Literal["mha", "mla", "dsa", "msa", "qwen_gdn",
  "inkling", "kimi_k3", "deepseek_v4"]`。
- `CachePoolSpec`（`:66`）：绑定一个 model 计算视图到 arena 所需的一切——`family` /
  `memory_plan: CacheMemoryPlan` / `layer_types` / `layer_group_ids` /
  `paged_cache_group_specs` / `state_field_dtypes` / `token_capacity` /
  `pool_options`。`pool_size`（`:83`）= `num_lcm_blocks * max_packing *
  logical_block_tokens`（child token 容量）；`layer_view(...)`（`:93`）为异构 draft 在
  同一 arena 上切 per-layer 元数据。
- `CacheSetup`（`:141`）：`spec` + `num_draft_layers` + `cache_budget_bytes` +
  `fixed_workspace_bytes`。

### 6.2 family→recipe 分派

- `_PREPARE_CACHE`（`:163`）：**family→recipe 注册表**（替代旧的 `lcm_family` 字符串
  判定）。mha/mla/dsa/msa → `partial(prepare_ordinary_cache, family=...)`；其余各自的
  prepare 函数。
- `prepare_cache_setup(*, family, ...)`（`:175`）：`create_attn_components` 调用的单一
  分派入口，`_PREPARE_CACHE[family](...)` 运行对应 recipe 返回 `CacheSetup`。

---

## 7. 物理存储 `kv_cache/base.py`（`CachePool`）

`python/tokenspeed/runtime/layers/attention/kv_cache/base.py:45`（旧 `LcmCachePool`）
——持有**一块** arena backing，按 plan 发 strided view。它是所有 KV pool 的物理底座。

- **backing**（`:271`-`277`）：`torch.zeros(self.plan.arena_bytes, dtype=torch.uint8,
  device=...)`——单块扁平 uint8，lazy 分配。
- **`field(field_id, dtype)`（`:180`）**：核心。按 plan 用 `as_strided` 发某 field 的
  strided view，起始偏移由 `_field_block_byte_offset`（`:279`）算（plane arena offset
  + parent 内偏移 + field 偏移）。
- **`_publish_runtime_contract`（`:139`）**：把每个 spec 的 `cache_blocks_per_lcm
  _block` 对齐到 plan 的 group packing，构造 `PagedCacheRuntimeContract`（`:96`
  一带调用）。
- **`pd_contract(group_specs)`（`:242`）**：给 PD 构造跨节点传输 contract，委托
  `runtime/pd/cache_protocol.py` 的 `build_lcm_pd_cache_contract`（`:244`/`:257`）。
- **`cache_transfer_layout`（`:318`）**：给 host L2 transfer 用，委托
  `runtime/cache/transfer/layout.py` 的 `layout_from_lcm_plan`。

具体计算接口 pool 继承 `CachePool`，把 buffer 从「自己 malloc」改成「绑到共享 arena
的 field view」，其余 attention ABI 不变：`kv_cache/mha.py`（`MHATokenToKVPool` /
`MHATokenToKVPoolMXFP8`）、`kv_cache/mla.py`（`MLATokenToKVPool`）、`kv_cache/dsa.py`、
`kv_cache/msa.py`、`kv_cache/hybrid_*.py`（Kimi-K3 KDA / Inkling / DeepSeek-V4 等
hybrid 变体）。

---

## 8. 与 registry / scheduler 集成

### 8.1 哪些模型走哪个 recipe（`registry.py`）

`create_attn_components`（`registry.py:763`）里由架构能力决定 `cache_family`
（`:839`-`864`）：

```python
use_cache_gdn     = is_hybrid_gdn and has_state     # Qwen3.5 GDN → "qwen_gdn"
use_cache_k3      = is_hybrid_mla_kda               # Kimi-K3 KDA → "kimi_k3"
use_cache_inkling = is_inkling                      # Inkling ShortConv → "inkling"
if is_deepseek_v4_model:  cache_family = "deepseek_v4"
elif use_cache_gdn:       cache_family = "qwen_gdn"
elif use_cache_k3:        cache_family = "kimi_k3"
elif use_cache_inkling:   cache_family = "inkling"
elif <MHAConfig>:         cache_family = "mha"
elif <MLAConfig>:         cache_family = "mla"
elif <DSAConfig>:         cache_family = "dsa"
elif <MSAConfig>:         cache_family = "msa"
else:                     cache_family = None        # → raise "No cache recipe ..."
```

> 旧的 `lcm_family` 字符串已彻底移除；现在是 (a) `recipes/setup.py:163` 的
> `_PREPARE_CACHE` 分派字典 + (b) 这段 `registry.py:839`-`864` 的 family 选择链。

`cache_family` 定好后：`prepare_cache_setup`（`registry.py` 里调用，
→ `recipes/setup.py:175`）跑 recipe 得 `CacheSetup`；`_validate_lcm_page_size`
（`registry.py:258`）校验逻辑页是 kernel 页的整数倍；`factory.create_cache_pool`
（`kv_cache/factory.py:12`）按 `spec.family` + config 类型造实际 pool。异构 draft
经 `spec.layer_view(...)` 共享同一 arena/plan。

### 8.2 scheduler 看到的几何

scheduler 通过 pool 的 `runtime_contract`（`num_lcm_blocks` / `group_page_counts`）
拿几何，由 `scheduler_utils.scheduler_cache_geometry_from_pool`（`:79`）翻译成
`SchedulerCacheGeometry`（`:71`），再由 `pool_to_paged_cache_groups`（`:248`）转成
C++ `PagedCacheGroupConfig` 列表喂给 `make_config`（`:198`）。调度器只在 parent
维度上分配 page，各 group 物理页由 packing 因子换算。

### 8.3 新页清零流程

`model_executor.zero_cache_pages(pages)`（`execution/model_executor.py:1177`）在
scheduler 发来「本步新拥有的页」（`execution_plan.pages_to_zero`）时被调，pool-agnostic
地转发到 pool 的 `zero_new_pages`（Mapping）/ `zero_pages`（page-id 列表），清零后返回
一个 CUDA completion event 供后续同步。（旧的 `zero_flat_cache_pages` 已被这条泛化
路径取代。）

---

## 9. 端到端数值示例（示意）

以 Qwen3.5 GDN 为例，直觉上一次分配长这样：

```
cache_budget_bytes = B
  │
  ├─ recipe 造 fields → solve_cache_layout：得到 packing，例如
  │     full_attention   packing = 1   (history K/V，每 parent 1 个 128-token page)
  │     linear_attention packing = N   (recurrent state 定长，塞得下 N 份)
  │
  ├─ lcm_block_bytes = lcm-align(sum(plane_bytes))       # 能被 1 和 N 整除
  ├─ num_lcm_blocks  = (B - workspace) // lcm_block_bytes - 1   # 减 1 = null parent
  │
  └─ arena = torch.zeros((num_lcm_blocks + 1) * lcm_block_bytes)  # +1 = null
        parent 0 ............ null（永不调度）
        parent 1 ┐
        parent 2 ├─ 每个 parent：full_attention 1 页 + linear_attention N 页
        ...      ┘   （物理上按 plane-major 交错，field view 通过 as_strided 取出）
```

调度器分配 parent 5 → full_attention 拿到 child page 5、linear_attention 拿到
child page `5*N .. 5*N+N-1`，两者的物理字节由各自 field 的 `as_strided` view 定位，
互不冲突且落在同一块 backing 内。

---

## 10. 文件索引

所有 recipe 文件都在
`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/` 下。

| 组件 | 位置 | 职责 |
|---|---|---|
| 几何引擎（LCM 数学） | `recipes/plan.py`（`solve_cache_layout:457`、`_solve_packing:363`、LCM 对齐 `:637`） | 纯整数几何：parent ↔ per-group child page 打包、plane overlay、LCM 对齐。 |
| `CacheMemoryPlan` / `CacheLayout` | `recipes/plan.py:67` / `:218` | 容量绑定 plan / 容量无关 layout（`with_num_lcm_blocks` 连接）。 |
| group spec / 页数 | `recipes/spec.py`（`PagedCacheGroupSpec:29`、`compute_paged_cache_group_page_counts:105`） | 几何 → 调度器契约（每组多少页、retention、transfer policy）。 |
| 运行时契约 | `recipes/cache_runtime.py`（`PagedCacheRuntimeContract:57`） | pool 发布给 scheduler/executor 的不变式。 |
| family 分派 | `recipes/setup.py`（`_PREPARE_CACHE:163`、`prepare_cache_setup:175`、`CachePoolSpec:66`） | family→recipe 注册表 + setup 结果容器。 |
| 各 family recipe | `recipes/ordinary.py`（mha/mla/dsa/msa）、`kimi_k3.py`、`deepseek_v4.py`(+`deepseek_v4_cache_spec.py`)、`inkling.py`、`qwen35.py` | 造 `CacheFieldSpec` + 选 packing 策略。 |
| 物理 arena | `kv_cache/base.py`（`CachePool:45`、`field:180`、arena `:271`、`pd_contract:242`） | 单块 arena backing，按 field 发 strided view，清零 + PD/transfer contract。 |
| 计算接口 pool | `kv_cache/{mha,mla,dsa,msa,hybrid_*}.py`（经 `factory.create_cache_pool:12` 造） | history K/V + state 共享 arena 的 attention ABI。 |
| registry 集成 | `layers/attention/registry.py`（family 门 `:839`-`864`、`_validate_lcm_page_size:258`） | cache_family 判定 + recipe 入口 + pool 构造。 |
| scheduler 几何 | `engine/scheduler_utils.py`（`scheduler_cache_geometry_from_pool:79`、`pool_to_paged_cache_groups:248`） | 从 pool contract 读几何 → C++ SchedulerConfig。 |
| 新页清零 | `execution/model_executor.py`（`zero_cache_pages:1177`） | `execution_plan.pages_to_zero` → pool `zero_new_pages`。 |

---

## 附：维护约定

本文与 `inference-flow.md` / `kvcache-management.md` 同规范：

- **第 5 行 `##### commit-id: <full-sha>`** 标记上次同步的 commit。更新前先
  `git log --oneline <sha>..HEAD` + `git diff --stat` 看改了什么，改完把 commit-id
  换成当前 HEAD 全 sha。
- **全文 `path:line` 锚点每个都要核对**（`sed -n 'Np'` / `grep -n` 逐个验证落在正确
  符号上）。recipe 系文件是漂移热点：`recipes/plan.py`、`recipes/setup.py`、
  `recipes/spec.py`、`recipes/ordinary.py`、`recipes/kimi_k3.py`、
  `recipes/deepseek_v4.py`、`kv_cache/base.py`、`kv_cache/factory.py`、`registry.py`。
- **注意 LCM 已是唯一 cache 路径**：不要把它写成「可选后端之一」；radix / flat-slab
  旧体系均已删除。
