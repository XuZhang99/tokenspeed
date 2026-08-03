# LCM 两级 KV-Cache 分配机制

##### commit-id: 8c4fa9765560d9e27f159b3968d0e716f87ae245

本文详细讲解 TokenSpeed 中 **LCM（two-level）KV-cache 分配体系**——它是 hybrid /
flat-state 模型（Qwen3.5 GDN、Kimi-K3 KDA、Inkling ShortConv）统一使用的 KV 内存
布局与分配机制，PR #804（`jenga two level allocation`）引入。本文可独立阅读，是
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
  `_LOGICAL_BLOCK_TOKENS = 128` 个逻辑 token（`lcm_setup.py:55`）。
- **下层单位 = per-group child page**：每个 cache group 在一个 parent 里按各自的
  **packing 因子 `cache_blocks_per_lcm_block`** 容纳固定整数个子 page。

分配**一个 parent** 就同时为**所有** group（history KV、recurrent/conv state、
draft history…）预留好各自数量的 page——调度器只需管理 parent 一个维度的整数，
每个 group 的物理页号由 parent id 乘以该 group 的 packing 因子直接算出。

### 1.3 名字由来："LCM" = 最小公倍数对齐

parent block 的字节大小必须能被**每个 group 的 packing 因子整除**（这样才能把
parent 均匀切成整数个 child page）。规划器通过对所有 packing 因子求
**最小公倍数（Least Common Multiple）**来定 parent 的对齐（`lcm_memory_plan.py:401`-`404`）：

```python
parent_alignment = alignment
for count in packing.values():
    parent_alignment = parent_alignment // math.gcd(parent_alignment, count) * count
lcm_block_bytes = _align_up(sum(plane_bytes.values()), parent_alignment)
```

即 `parent_alignment = lcm(alignment, *packing.values())`。这个 LCM 对齐正是整个
体系名字的来源。

> **旧体系已删除**：flat-slab 时代的 `flat_hybrid.py` / `flat_state_slabs.py` /
> `hybrid_cache_plan.py` / `flat_memory_plan.py` 已被 LCM 取代
> （`refactor(scheduler): make two-level cache unconditional`）。

---

## 2. 核心概念与术语

| 术语 | 含义 |
|---|---|
| **parent LCM block** | arena 的分配/调度单位，固定 128 逻辑 token。共 `num_lcm_blocks` 个（不含 null）。 |
| **cache group** | 一组共享同一 retention/packing 的层的集合，如 `full_attention` / `linear_attention` / `kvconv` / `hiddenconv`。 |
| **packing 因子** | `cache_blocks_per_lcm_block`，一个 parent 里某 group 塞几个 child page。 |
| **child page / cache block** | group 的最小物理页；一个 parent 里某 group 有 `packing` 个。 |
| **plane** | arena 内的一段连续区域，同一 plane 被多个 group 的同名 field **overlay**（错位复用）共享。plane-major 布局。 |
| **field** | 一个具体张量视图的几何描述（layer 的 k / v / latent_kv / conv_state 等），归属某 group + 某 plane。 |
| **null LCM block（parent 0）** | 永不调度的哨兵 parent，backing 逻辑 null page 0。`arena_bytes` 含它，`num_lcm_blocks` 不含。 |
| **logical_block_tokens** | 逻辑页 token 数，固定 128（`_LOGICAL_BLOCK_TOKENS`）。 |

**两级映射直觉**：某 group（packing = K）的 child page 号 = `parent_id * K + slot`
（slot ∈ [0, K)）；每组 `page_count = 1 + num_lcm_blocks * K`（`lcm_memory_plan.py:440`，
`1 +` 是 null parent 的哨兵页）。

---

## 3. 几何层 `LcmMemoryPlan`

`python/tokenspeed/runtime/configs/lcm_memory_plan.py`——**纯整数几何，不 import
torch**。它只负责回答「arena 有多大、每个 field 落在哪个字节偏移、page stride 多少」。

### 3.1 数据类

- `LcmGroupLayout`（`:36`）：`group_id` + `cache_blocks_per_lcm_block`（packing）+
  `page_count`。
- `LcmFieldSpec`（`:43`）：**输入**配方——`group_id` / `field_id` / `plane_id` /
  `shape` / `element_size` / `exact_page_stride`。`payload_bytes = prod(shape) *
  element_size`。
- `LcmPlaneLayout`（`:59`）：`plane_id` + `bytes_per_lcm_block` + `arena_offset_bytes`。
- `LcmFieldLayout`（`:66`）：**输出**——在 `LcmFieldSpec` 基础上加算好的
  `field_offset_bytes` + `page_stride_bytes`。
- `LcmMemoryPlan`（`:81`）：最终产物，含 `groups` / `planes` / `fields`，以及
  `arena_bytes`（`:95`，`= (num_lcm_blocks + 1) * lcm_block_bytes`）和
  `group()` / `field()` / `plane()` 查询。

### 3.2 `exact_page_stride`：两类 kernel

`LcmFieldSpec.exact_page_stride`（`:51`）区分 field 的 kernel 对 stride 的假设：

- `True`（默认）：kernel 按**隐式 payload-sized stride** 走页——page stride 必须
  精确等于 `payload_bytes`，否则读错行。history K/V、MLA latent、scale field 属此类。
- `False`：kernel 在 launch 时读取运行时 `stride(0)`，可以吸收其它 field 在 plane
  里留下的 slack。conv/ssm state、Inkling 的 kvconv/hiddenconv 属此类（如
  `lcm_layouts.py:198` 注释：`causal_conv1d reads stride(0) at launch`）。

### 3.3 规划入口 `plan_lcm_fields(...)`（`:217`）

规划一个 **plane-major** arena，field 在 cache group 之间 overlay。关键行为：

1. **容量来源二选一**（`:235`）：`budget_bytes`（推导 `num_lcm_blocks`，`:412`）
   **或** `num_lcm_blocks`（固定，用于 colocated draft 保持 parent 几何一致）。
2. **packing 推导**：
   - `_packing_by_group_ratio`（`:185`）：按各 group 原始字节比给初值。
   - `_solve_packing`（`:121`）：对**同一 plane 上 exact 字节**建立 group 间比例
     约束（并查集 + 分数化简），推出让各 group page stride 精确一致的整数 packing。
   - 显式 `cache_blocks_per_lcm_block`（`:288`）覆盖以上，供 draft 复用 target 几何。
3. **flexible group 吸收 slack**（`:319`-`376`）：非 exact 的 group 在其占用的每个
   plane 上，取「fixed field 定死的 plane 字节 ÷ payload」的最大可整除 count。
4. **plane / parent 字节**：`plane_bytes`（`:378`）逐 plane 对齐；`lcm_block_bytes`
   对所有 packing 求 LCM 对齐（`:401`-`404`，见 §1.3）。
5. **校验（fail loud）**：
   - `_check_exact_page_strides`（`:193`）：exact field 的 `plane_bytes // packing`
     必须 == `payload_bytes`，否则报错并提示是哪个 field 把 plane 撑大了。
   - padding fraction ≤ `max_padding_fraction`（`:426`）。
   - kernel page id 不超 `_MAX_KERNEL_PAGE_ID = 2^31 - 1`（`:431`）。
   - `lcm_block_bytes` 不超 `_MAX_LCM_BLOCK_BYTES = 2^63 - 1`。
6. **产出**：每组 `page_count = 1 + num_lcm_blocks * count`（`:440`）；plane 按
   `(num_lcm_blocks + 1) * nbytes` 依次排 `arena_offset`（`:445`-`449`）；每个 field
   的 `page_stride_bytes = plane_bytes[plane] // packing[group]`（`:461`）。

---

## 4. 布局 recipe `lcm_layouts.py`

`python/tokenspeed/runtime/configs/lcm_layouts.py`——把每个 model family 的层结构翻译
成一串 `LcmFieldSpec`，喂给 `plan_lcm_fields`。每个 recipe 用 `occurrences` 计数把
同 group 的层映射到不同 `slot.N` / `unit.N` plane，实现 plane overlay。

| recipe | 行 | 用途 | 关键 field |
|---|---|---|---|
| `mla_history_lcm_fields` | `:28` | MLA latent history（Kimi-K3 draft、DeepSeek 系 draft） | `layer.{id}.latent_kv`，shape `(P, 1, latent_width)` |
| `draft_history_lcm_fields` | `:55` | MHA draft 模型的 history K/V(+scale) | `layer.{id}.k` / `.v` / `.k_scale` / `.v_scale` |
| `qwen_gdn_lcm_fields` | `:154` | Qwen3.5：full-attn history + GDN recurrent | history `.k`/`.v`；linear `.ssm`（exact）/`.conv`（flexible） |
| `inkling_lcm_fields` | `:224` | Inkling：attention page + ShortConv checkpoint | `.k`/`.v` + `kvconv`/`hiddenconv` group（均 flexible） |
| `kimi_k3_lcm_fields` | `:357` | Kimi-K3：MLA latent + KDA state（24 plane 共享） | full `.latent_kv`；linear `.conv_state`/`.recurrent_state`（均 flexible） |

**要点**：linear-attention 层的 conv/ssm state 用 `exact_page_stride=False`，让它们
吸收同 plane 上 history field 留下的 slack；MXFP8 变体额外挂 `.k_scale`/`.v_scale`
field（`lcm_layouts.py:124`-`150`、`:327`-`353`）。

---

## 5. Setup 编排 `lcm_setup.py`

`python/tokenspeed/runtime/layers/attention/lcm_setup.py`——从单个
`cache_budget_bytes` 同时给 target + draft arena 定尺寸并造 pool。

### 5.1 产物数据类

- `LcmPoolSpec`（`:60`）：绑定一个 model 计算视图到 arena 所需的一切
  （`memory_plan` / `layer_types` / `layer_group_ids` / `state_field_dtypes` /
  `token_capacity` / `extra_paged_groups`）。`pool_size`（`:72`）=
  `num_lcm_blocks * max_packing * logical_block_tokens`（child token 容量）。
- `LcmSetup`（`:84`）：`target` + 可选 `draft` + `cache_budget_bytes` +
  `fixed_workspace_bytes`。

### 5.2 分派入口 `prepare_lcm_setup(...)`（`:578`）

按 `family` 分派：`kimi_k3` → `_prepare_kimi_k3`（`:244`），其余
（`qwen_gdn` / `inkling`）→ `_prepare_mha`（`:381`）。

**`_prepare_mha`（MHA 家族）流程**：
1. 按 family 造 `fields` + `state_dtypes` + `fixed_workspace_bytes`（GDN verify 行、
   Inkling ShortConv rolling/verify workspace，`:432`-`472`）。
2. 用 `budget_bytes` 先规划一个 `reference_plan`（`:474`）拿到 packing 与
   `lcm_block_bytes`。
3. 若有 draft：`_draft_history_fields`（`:189`）造 draft field，复用 target packing，
   单独规划 `draft_parent_bytes`（`:505`）。
4. `num_lcm_blocks = (budget - workspace) // (target_parent + draft_parent) - 1`
   （`:514`-`516`），再受 `token_limit`（`_token_limit`，`:97`）夹逼（`:523`-`531`）。
5. 用**固定 `num_lcm_blocks` + 显式 packing** 重新规划 target/draft plan
   （`:533`、`:554`），保证 target 与 draft parent 几何一致。

**`_prepare_kimi_k3` 流程**（`:244`）：类似，但走 `kimi_k3_cache_spec` 的
`plan_kimi_k3_lcm_cache` / `kimi_k3_lcm_blocks_needed` / `token_capacity_for_lcm_pool`
做 token 容量推导，draft 用 `mla_history_lcm_fields` + 固定 packing `{full_attention: 12}`。

### 5.3 造 pool `create_lcm_pool(...)`（`:613`）

按 config 类型分派：`MHAConfig` → `LcmMHATokenToKVPool`（MXFP8 时用
`LcmMHATokenToKVPoolMXFP8`）；`MLAConfig` → `LcmMLATokenToKVPool`。把 `memory_plan` /
`layer_group_ids` / `state_field_dtypes` / `token_capacity` 等透传进去。

---

## 6. 物理存储 `LcmCachePool`

`python/tokenspeed/runtime/layers/attention/kv_cache/lcm.py:31`——持有**一块** arena
backing，按 plan 发 strided view。它是 flat-state pool 的物理底座。

- **backing**（`:36`）：`torch.zeros(plan.arena_bytes, dtype=torch.uint8, device=...)`
  ——单块扁平 uint8。
- **`field(field_id, dtype)`（`:39`）**：核心。按 plan 用 `as_strided` 发某 field 的
  strided view，shape `(page_count, *field.shape)`，page stride =
  `page_stride_bytes // element_size`，起始偏移由 `_field_page_byte_offset` 算
  （`:101`，含 plane arena offset + parent 内偏移 + field 偏移）。缓存进 `_fields`，
  重复取会校验 dtype 一致（`:42`）。
- **`zero_pages(page_ids_by_group)`（`:71`）**：把指定 group 的指定页对应的字节段
  收集成 `(offset, len)` segment 列表，交给 `zero_byte_segments` kernel 清零。
- **`pd_contract(group_specs)`（`:80`）**：给 FlatKV PD 构造跨节点传输 contract
  （委托 `build_lcm_flatkv_pd_contract`），要求所有 field 都已绑定运行时 dtype。

---

## 7. 计算接口 `LcmMHATokenToKVPool` / `LcmMLATokenToKVPool`

两个 pool 分别继承普通的 `MHATokenToKVPool` / `MLATokenToKVPool`，把 buffer 从
「自己 malloc」改成「绑到共享 arena 的 field view」，其余 attention ABI 不变。

### 7.1 `LcmMHATokenToKVPool`（`lcm_mha.py:44`）

- 构造时置 `flat_kv_requires_page_zeroing = True`（`:67`），声明「复用页必须先清零」
  （因为 state 字节和 KV 字节 alias 在同一 arena）。
- **`_create_lcm_buffers`（`:141`）**：逐层——
  - state 层（`linear_attention`）：绑 `(conv, ssm)` 到 `_state_buffers_by_layer`，
    并校验 ssm plane 无 padding（GDN decode ABI 要求连续，`:172`）。
  - history 层：`k_buffer/v_buffer` 绑到 `layer.{id}.k`/`.v` field 的
    `view(-1, head_num, head_dim)`，校验无 page 间 padding（`:184`）。
- **`_publish_paged_cache_groups`（`:89`）**：调
  `paged_cache_spec.publish_paged_cache_groups(..., cache_blocks_per_lcm_block=...)`
  上报 group spec 与每组 `page_count`；PD 开启时给 state 组打 `latest_snapshot`、
  history 组打 `full_suffix` transfer policy（`:121`-`130`）。
- **`zero_new_pages(new_page_ids)`（`:244`）**：转发到 `lcm_pool.zero_pages`，供
  scheduler 清零新分配的组页。
- MXFP8 变体 `LcmMHATokenToKVPoolMXFP8`（`:258`）再绑 `.k_scale`/`.v_scale` field。

### 7.2 `LcmMLATokenToKVPool`（`lcm_mla.py:40`）

- 构造时同样 `publish_paged_cache_groups`（`:77`）+ 建 `FlatPagedCacheRuntimeContract`
  （`:109`）。
- **`_create_lcm_buffers`（`:121`）**：MLA latent 层 `kv_buffer` 绑
  `layer.{id}.latent_kv` field（`:160`）；KDA state 层绑
  `(conv_state, recurrent_state)`（`:150`-`158`）——这正是 Kimi-K3 布局。
- `get_component(layer_id, name)`（`:194`）：按 `latent_kv` / `conv_state` /
  `recurrent_state` 返回对应 view，供 KDA kernel 取用。

---

## 8. 与 registry / scheduler 集成

### 8.1 哪些模型走 LCM（`registry.py`）

`create_attn_components` 里根据模型能力决定 `lcm_family`（`:1027`-`1035`）：

```python
use_lcm_gdn     = is_hybrid_gdn and has_flat_state and flat_kvcache   # Qwen3.5 GDN
use_lcm_k3      = is_hybrid_mla_kda and flat_kvcache                   # Kimi-K3 KDA
use_lcm_inkling = is_inkling and flat_kvcache                          # Inkling ShortConv
lcm_family = "qwen_gdn" if use_lcm_gdn else "kimi_k3" if use_lcm_k3 \
             else "inkling" if use_lcm_inkling else None
```

`lcm_family is not None` 时（`:1209`）：先 profile 可用显存拿 `cache_budget_bytes`，
调 `prepare_lcm_setup`（`:1218`），校验逻辑页大小，再 `create_lcm_pool` 造实际 pool。
返回 8 元组第 8 个 `cache_storage`（`_cache_storage_report`，`registry.py:195`）报告
实际分配字节 / token 容量 / 几何，随 scheduler ready payload 上报。

### 8.2 scheduler 看到的几何

scheduler 通过 pool 的 `num_lcm_blocks`（`lcm_mha.py:198` / `lcm_mla.py:171`）与
`runtime_contract` 得到几何（`scheduler_utils.py:88`-`113`）：`num_device_pages =
num_lcm_blocks + 1`（含 null parent），`num_usable_pages = num_lcm_blocks`。调度器只在
parent 维度上分配，无需关心各 group 的物理页——那由 packing 因子换算。

### 8.3 新页清零流程

`model_executor.zero_flat_cache_pages(pages)`（`model_executor.py:1499`）在 scheduler
发来「本步新拥有的页」时被调：`pages` 是 `{group_id: [page_id]}` 的 Mapping 且 pool
有 `zero_new_pages` 就走 LCM 这条（`:1505`）；若 pool 声明了
`flat_kv_requires_page_zeroing` 却没实现 sanitizer 则 fail loud（`:1528`）。清零后在
CUDA 上返回一个 completion event 供后续同步。

---

## 9. 端到端数值示例（示意）

以 Qwen3.5 GDN 为例，直觉上一次分配长这样：

```
cache_budget_bytes = B
  │
  ├─ 规划 reference_plan：得到 packing，例如
  │     full_attention  packing = 1   (history K/V，每 parent 1 个 128-token page)
  │     linear_attention packing = N  (recurrent state 定长，塞得下 N 份)
  │
  ├─ lcm_block_bytes = lcm-align(sum(plane_bytes))  # 能被 1 和 N 整除
  ├─ num_lcm_blocks  = (B - workspace) // parent_bytes - 1   # 减 1 = null parent
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

| 组件 | 位置 | 职责 |
|---|---|---|
| `LcmMemoryPlan` / `plan_lcm_fields` | `configs/lcm_memory_plan.py:81` / `:217` | 纯整数几何：parent ↔ per-group child page 打包、plane overlay、LCM 对齐。 |
| layout recipes | `configs/lcm_layouts.py` | 各 model family 的 `LcmFieldSpec` 配方（mla/draft/qwen_gdn/inkling/kimi_k3）。 |
| `prepare_lcm_setup` / `create_lcm_pool` | `layers/attention/lcm_setup.py:578` / `:613` | 从 budget 给 target+draft 定尺寸并造 pool。 |
| `LcmCachePool` | `layers/attention/kv_cache/lcm.py:31` | 单块 arena backing，按 field 发 strided view，清零 + PD contract。 |
| `LcmMHATokenToKVPool` | `kv_cache/lcm_mha.py:44` | MHA 计算接口：history K/V + GDN/ShortConv state 共享 arena。 |
| `LcmMLATokenToKVPool` | `kv_cache/lcm_mla.py:40` | MLA 计算接口：latent KV + KDA state 共享 arena（Kimi-K3）。 |
| registry 集成 | `layers/attention/registry.py:1027` / `:1209` | `lcm_family` 判定 + LCM setup 入口 + `cache_storage` 报告。 |
| scheduler 几何 | `engine/scheduler_utils.py:88` | 从 pool 读 `num_lcm_blocks` → `num_device_pages = +1`。 |
| 新页清零 | `execution/model_executor.py:1499` | `zero_flat_cache_pages` → pool `zero_new_pages`。 |

---

## 附：维护约定

本文与 `inference-flow.md` / `kvcache-management.md` 同规范：

- **第 2 行 `##### commit-id: <full-sha>`** 标记上次同步的 commit。更新前先
  `git log --oneline <sha>..HEAD` + `git diff --stat` 看改了什么，改完把 commit-id
  换成当前 HEAD 全 sha。
- **全文 `path:line` 锚点每个都要核对**（`sed -n 'Np'` / `grep -n` 逐个验证落在正确
  符号上）。LCM 系文件是漂移热点：`lcm_setup.py`、`lcm_memory_plan.py`、
  `lcm_layouts.py`、`lcm.py`、`lcm_mha.py`、`lcm_mla.py`、`registry.py`。
