# KV Cache 管理机制
##### commit-id: d50bb481505c229f93594e56d5e1f8a772876451

本文全面介绍 TokenSpeed 的 KV-cache 管理机制：从物理显存布局、逻辑分块分配、
prefix 复用/驱逐，到 GPU→host→外部存储的分层内存，以及 Prefill/Decode
分离（PD disaggregation）下的 KV 跨节点传输。可与仓库根目录的 `inference-flow.md`
（讲一次生成请求的端到端流程）配合阅读——本文是它「KV / 内存」维度的展开。

## 总览

KV-cache 管理分成几个相对独立又相互咬合的层次：

```text
┌─ 分配决策（C++ scheduler） ─────────────────────────────────────────┐
│  tokenspeed_scheduler.Scheduler：按 paged_cache_groups 做逻辑分块    │
│  分配、prefix 复用、驱逐，产出 execution_plan（含要清零的页、cache  │
│  op、forward op）。Python 侧只提供几何/容量，不持有分配算法。         │
└──────────────────────────────────────────────────────────────────┘
        │ bind_paged_cache_scheduler / submit_requests / next_execution_plan
        ▼
┌─ 物理 KV 存储（Python token-to-KV pool） ──────────────────────────┐
│  BaseTokenToKVPool 家族：MHA / MLA / DSA / DeepSeekV4 / MSA         │
│  + flat-state 模型的 LCM two-level arena（LcmMHA / LcmMLA）         │
│  持有真正的 GPU KV buffer，按 loc(物理 slot) 读写。                  │
└──────────────────────────────────────────────────────────────────┘
        │ 逻辑页 → kernel 页展开（page_table.expand_page_table）
        ▼
┌─ 分层内存（可选，非 flat 构建） ────────────────────────────────────┐
│  MemoryExecutor：L1 GPU radix prefix tree（PrefixCache）            │
│                → L2 host DRAM（HostExecutor + HostKVCache）         │
│                → L3 外部存储（StorageExecutor + Mooncake）          │
│  flat 构建走精简版 FlatMemoryExecutor（只有 host mirror，无 L3）。   │
└──────────────────────────────────────────────────────────────────┘
        │ disaggregation_mode != "null"
        ▼
┌─ PD 跨节点 KV 传输 ────────────────────────────────────────────────┐
│  create_kv_transfer → DisaggPrefill/DecodeExecutor（Mooncake RDMA）│
│  prefill 节点算完 KV 发给 decode 节点；first-token 随控制面回传。    │
└──────────────────────────────────────────────────────────────────┘
```

一个核心区分：**分配算法在 C++ scheduler 里**，Python 侧的 pool 只负责「按物理
slot 索引读写 KV bytes」并向 scheduler 上报几何（每个 group 多少页、页多大）。
遗留的 radix 路径另有一个纯 Python 的 `KVAllocator`（见第 3 节），但生产热路径
（尤其 flat 构建）由 C++ scheduler 分配。

## 1. 关键抽象与术语

三个层层嵌套的粒度，务必分清（定义见
`python/tokenspeed/runtime/configs/paged_cache_spec.py:23`-`42`）：

- **cache group（缓存组）**：一批共享同一保留策略/几何的层的逻辑分区。由
  `group_specs_from_layer_types(...)` 按「不同 `(retention, window)` 各成一组」
  推导（`paged_cache_spec.py:531`）。同一模型可并存多组，无固定上限：
  - **history KV 组**：`full_attention` → `full_history` 保留；
    `sliding_attention` → `sliding_window`（`paged_cache_spec.py:96`-`110`）。
  - **recurrent/conv state 组**：`linear_attention` → `family="state"`
    （Qwen3.5 GDN / Kimi-K3 KDA / Inkling sconv），按位置用
    `split_recurrent_state_groups(...)` 拆分（`paged_cache_spec.py:512`）。
  - **draft/spec 组**：drafter 复用 full-attention 组的表。
  - **模型自声明的 extra 组**（如 Inkling 的 paged sconv checkpoint）经
    `extra_groups` 合并（`paged_cache_spec.py:628` 的 `publish_paged_cache_groups`）。
- **page（页）**：一个 group 的分配单位，容纳 `rows_per_page * entry_stride_tokens`
  个 token；每组可有自己的 `block_size`（`paged_cache_spec.py:38`）。
- **block（物理块 / CacheBlock）**：物理 backing 的单位；`cache_blocks_per_lcm_block`
  （`paged_cache_spec.py:40`）把若干物理子 block 打包进一个共享的 LCM parent
  block（见第 5 节）。

scheduler 自己的 `block_size` 是所有 group block_size 的 **gcd**
（`scheduler_utils.py:139` 的 `resolve_scheduler_block_size`，gcd 在 `:144`），
它是 prefix 哈希的粒度，不等于 KV 页几何。

`PagedCacheGroupSpec`（`paged_cache_spec.py:28`）的字段：`group_id`、`retention`
（`full_history` / `sliding_window`）、`rows_per_page`、`entry_stride_tokens`、
`sliding_window_tokens`、`family`（`history` / `state`，默认 `history`，`:36`）、
`block_size`（`:38`）、`cache_blocks_per_lcm_block`（`:40`）、`transfer_policy`
（`full_suffix` / `latest_snapshot`，PD 用，`:42`）。

`BaseTokenToKVPool`（`python/tokenspeed/runtime/layers/attention/kv_cache/base.py:37`）
是所有物理 KV pool 的基类，关键类属性：

- `paged_cache_group_specs` / `paged_cache_group_page_counts`（`base.py:40`-`41`）：
  pool 向 scheduler 上报的 group 列表和每组页数。
- `supports_hierarchical_kv_cache`（`base.py:42`，默认 `True`）：是否支持 L2/L3
  分层——它决定 event_loop 是否给这个 pool 建 `MemoryExecutor`（见第 7 节）。
  slab/LCM 布局会置 `False`。
- `flat_kv_requires_page_zeroing`（`base.py:46`，默认 `False`）：只有把 recurrent
  state bytes 和 KV 混叠在同一 slab 的 flat pool 才需要在复用时清零物理页；纯
  attention pool 复用页会被覆写、tail 不会被读，无需 sanitize（见第 6/8 节）。
- 抽象接口：`cell_size()`（`:77`）、`get_key_buffer/get_value_buffer/get_kv_buffer`
  （`:123`-`130`）、`set_kv_buffer`（`:132`）、`get_cpu_copy/load_cpu_copy`
  （`:141`-`147`，L2 offload 用）、`get_contiguous_buf_infos`（`:155`，PD 用）、
  `clear_kv_buffers`（`:91`，sleep/wake 后清零重映射页）。

## 2. Paged block 分配层（Python 元数据侧）

### 2.1 `KVAllocator`（radix 路径的纯元数据分配器）

`python/tokenspeed/runtime/cache/allocator.py:30`。它只管 token-slot / block-table
记账，**不持有物理 KV bytes**（docstring `:31`），故跨内存后端通用。这是遗留/radix
paged 分配器；flat-scheduler 路径改在 C++ 里分配。它被
`PrefixCache`（`prefix_cache.py:80` 的 `token_to_kv_pool_allocator` 字段）持有。

- `__init__`（`:37`）：建 `free_slots`、`token_slot_refs`、`req_to_page` /
  `req_to_page_cpu`、`max_page_num = ceil(max_context_len/page_size)`。
- `available_size()`（`:61`）：空闲页数。
- `alloc(req_pool_index, need_size, alloced_len)`（`:64`）：先填最后一页的尾部
  （`:65`），再从 `free_slots` 前部切新页（`:98`），写入 `req_to_page[...]`
  （`:101`），返回展平的物理 slot 索引 `kv_loc`；OOM 或页越界返回 `None`。
- `free_with_diff(new_prefix_page_ids, old_page_ids)`（`:138`）：只释放与复用
  prefix 不同的页——这是 prefix caching 的释放原语。
- `free`（`:161`）/ `free_group_end`（`:181`）：立即或批量把页退回 `free_slots`
  并减 `token_slot_refs`。
- `clear`（`:198`）：**page 0 保留作 padding**（`:199`），重建 `token_slot_refs`
  和 `req_to_page`（`[max_batch_size, max_page_num]` int32）。

空闲页跟踪 = 1-D int32 `free_slots`（页 id）；per-slot refcount 在 `token_slot_refs`。

### 2.2 `ReqToTokenPool`（请求 → token 位置表）

`python/tokenspeed/runtime/cache/req_to_token_pool.py:48`。映射 req-pool index →
物理 token 位置（token 粒度，区别于页表）。

- `ReqToTokenPoolInfo`（`:40`）dataclass：`verified_len` / `alloced_len` /
  `alloced_slots`，服务 chunked prefill。
- `__init__`（`:51`）：`req_to_token = zeros[size, max_context_len]` int32
  （`:68`），tag `"kv_cache"`；另有 `verified_lens[size]`（有效历史长度）、
  `alloced_lens[size]`（可超过 verified）+ pinned CPU 镜像。
  `free_slots = range(size)[1:]`——**slot 0 保留 padding**（`:78`）。
- `set_req_pool_info`（`:80`）、`write`（`:86`）、`alloc(need_size)`（`:92`，OOM 返回
  `None`）、`free`（`:108`）、`clear`（`:116`，flush_cache 用，保留 slot 0）。

### 2.3 逻辑页 → kernel 页展开

`python/tokenspeed/runtime/layers/attention/page_table.py`。纯 helper，把
scheduler 的「逻辑」页 id 转成具体 attention kernel 消费的更小页：

- `_page_ratio`（`:22`）：逻辑页必须是 kernel 页的正整数倍。
- `expand_page_table(...)`（`:35`）：2-D `[rows, cols]` 块表进 → 展开的 2-D 表出；
  `ratio==1` 直接切/拷（`:55`），否则每个逻辑页 `p` 广播成
  `[p*ratio, p*ratio+1, ...]`（`:87`）。

### 2.4 Python ↔ C++ scheduler 的分配绑定

C++ 的 `Scheduler` / `SchedulerConfig` / `RequestSpec` / `PagedCacheGroupConfig`
来自编译的 `tokenspeed_scheduler` 扩展（`scheduler_utils.py:31`、`event_loop.py:33`）。
`EventLoop.__init__` 里的绑定顺序：

1. `event_loop.py:395`：`paged_cache_groups = pool_to_paged_cache_groups(pool)`
   把 Python `PagedCacheGroupSpec`（或 `runtime_contract`）翻译成 C++
   `PagedCacheGroupConfig`（`scheduler_utils.py:259`，`total_pages` 从 page_counts
   取、`block_size` 在 `:309`）。
2. `event_loop.py:431`：`validate_flat_scheduler_config(...)` fail-fast
   （`paged_cache_spec.py:207`，拒绝 table-blind 后端配多 group pool、零 group、
   contract pool 的 family 覆盖缺口）。
3. `event_loop.py:442`：`prefix_cache_adjunct = pool_to_prefix_cache_adjunct_spec(...)`
   （`scheduler_utils.py:314`）；`event_loop.py:461` 的 `make_config(...)`
   （`scheduler_utils.py:200`）构造 `SchedulerConfig`，写 `cfg.block_size`（gcd，
   `scheduler_utils.py:227`）、`cfg.paged_cache_groups`、`cfg.prefix_cache_adjunct`
   （`event_loop.py:482`）。
4. `event_loop.py:506`：**`self.scheduler = Scheduler(scheduler_cfg)`**（C++ ctor）。
5. `event_loop.py:507`：**`pool.bind_paged_cache_scheduler(self.scheduler)`**（hook
   在 `base.py:87`，DeepSeek-V4 等有 override）。

每步交互：`self.scheduler.submit_requests(specs)` 入队已 admit 的请求
（`event_loop.py:763` / `:1276`），**`execution_plan =
self.scheduler.next_execution_plan()`**（`event_loop.py:1734` / `:1890`）是 C++
真正做块分配、返回执行计划（要清零的页、cache op、forward op）的地方；
`advance_forward(self.scheduler, request_changes)`（`event_loop.py:1665` 等）推进
scheduler FSM。

若开 prefix caching，还会先用 `aligned_max_scheduled_tokens(...)`
（`scheduler_utils.py:148`）把 `chunked_prefill_size` floor 到 state-snapshot 页
对齐（PR #830）。

### 2.5 每步 forward 如何更新 `req_to_page` / 块表

`req_to_page` 在 `ModelExecutor.__init__` 建为
`zeros[max_req_pool_size+1, max_num_pages_per_req]` int32
（`model_executor.py:345`）。每步：

- **`ModelExecutor.update_block_table(forward_op)`**（`model_executor.py:1489`）调
  `update_block_table` kernel driver，用 scheduler 发来的页刷新切片更新
  `self.req_to_page`。
- 底层 Triton kernel 是 `update_req_to_page`（`cache_loc_kernel.py:85`，kernel 在
  `:35`），把 `new_occupied_pages` 写进每个请求的行；`compute_out_cache_loc`
  （`cache_loc_kernel.py:217`，kernel `:127`）算物理写位置：
  `page_id = req_to_pages[req_pool_idx, position // page_size];
  out_cache_loc = page_id*page_size + position%page_size`。
- flat+spec 下，`_mirror_flat_full_table_into_req_to_page(...)`
  （`model_executor.py:698`，调用点 `:1823`）把 full-attention 组的 flat 块表原样
  `index_copy_` 进 `req_to_page` 行，让 drafter / 非-flat 回退仍可用。

## 3. KV pool 家族（物理存储）

所有 pool 继承 `BaseTokenToKVPool`。共同点：显存布局是
`(size + slot/page_size, ...)` 的大 buffer，**page/slot 0 是零初始化的 dummy 页**
——padded token 写到那里。`set_kv_buffer(layer, loc, cache_k, cache_v)` 按 `loc`
（物理 slot 索引）散写，`get_key_buffer/get_value_buffer` 返回给 attention backend
消费（backend delegate 见 `paged_attention.py:65` 的 `forward`，委托到
`ctx.attn_backend.forward(...)` 在 `:86`）。

### 3.1 `MHATokenToKVPool`

`python/tokenspeed/runtime/layers/attention/kv_cache/mha.py:53`。标准 MHA/GQA：
per-layer `k_buffer` / `v_buffer`，形状 `(size + slot_tokens, head_num, head_dim)`
（`_create_kv_buffers` 的 `alloc()` 在 `:264`）。要点：

- **hybrid slab 布局**（`:271`）：混合层（如 GPT-OSS full+sliding、Qwen3.5 GDN）时，
  按 `hybrid_slab_group_size` 把每 group 第 i 层绑到同一个 slab（`_slab_pair_index`
  在 `:194`），异构层通过 per-layer view 复用同一块 bytes；此模式下
  `supports_hierarchical_kv_cache=False`（`:282`），且 `_check_slab_guards`（`:217`）
  拒绝与 PD 不兼容的组合。
- **异构 KV head（#647）**：`_layer_row_view`（`:494`）/ `_layer_heads_per_rank`
  （`:509`）把 byte-uniform slot 按每层 head 数重解释成不同行数。
- **conv slot view**（`:522` `conv_slot_view` / `:557` `kvconv_slot_views_for_layer`）：
  给 Inkling 之类的 paged-conv 复用 KV slot 的前缀 bytes 做卷积列视图。
- `set_kv_buffer`（`:608`）用 `store_kv_cache` triton kernel 散写；
  `_publish_paged_cache_groups`（`:169`）在构造时调
  `paged_cache_spec.publish_paged_cache_groups` 上报 group。
- `get_contiguous_buf_infos`（`:408`）/ `get_layerwise_buf_info_offsets`（`:435`）
  给 PD 传输提供 per-layer buffer 指针。
- `MHATokenToKVPoolMXFP8`（`:639`）：存 MXFP8 block-scaled FP8（data + UE8M0
  scale），`set_kv_buffer`（`:788`）期望预量化输入 + per-token scale，
  `quantize_and_set_kv_buffer`（`:843`）融合量化+散写。

### 3.2 `MLATokenToKVPool`

`python/tokenspeed/runtime/layers/attention/kv_cache/mla.py:50`。MLA 只存压缩的
latent KV，单 `kv_buffer` per layer，形状 `(size + page_size, 1, kv_lora_rank +
qk_rope_head_dim)`（`_create_buffers` 在 `:123`）。两种量化：

- 默认：单 tensor，`set_mla_kv_buffer`（`:341`）用 `set_mla_kv_buffer_triton`
  分别写 nope / rope 部分，`get_mla_kv_buffer`（`:383`）读回。
- `quant_method == "per_token_head"`（`:126`）：`kv_buffer` 是 `(k_lora, k_scale,
  k_rope)` 三元组，per-token FP8 量化。

KDA state 层的 `kv_buffer[layer_id]` 为 `None`（`get_key_buffer` 在 `:290` 对 state
层 raise）——Kimi-K3 就是 MLA 层 + KDA state 层混合。

### 3.3 `DSATokenToKVPool`（sparse MLA，GLM-5）

`python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py:37`，继承
`MLATokenToKVPool`。在 MLA latent 之外，额外为 lightning-indexer 存一份
`index_k_buffer`（`:49`，形状 `(size + page_size, index_k_row_bytes)` uint8，FP8
量化，group size 128）。`_get_page_size_bytes`（`:64`）把 indexer bytes 计入页大小。

### 3.4 `MSATokenToKVPool`（MiniMax M3）

`python/tokenspeed/runtime/layers/attention/kv_cache/msa.py:13`，继承
`MHATokenToKVPool`。MiniMax M3 的 dense/sparse 逐层混合，但 **KV cache 仍是单个
full-history 组**（sparse 由 compute 侧的 lightning-indexer 处理，不改 KV 几何，见
`configs/msa.py`）。

### 3.5 `DeepseekV4TokenToKVPool`

`python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py:755`，直接继承
`BaseTokenToKVPool`。DeepSeek-V4 最复杂：一个 pool 里并存多种 buffer——SWA KV、
per-ratio 压缩 KV（`v4.c{ratio}a.compressed_kv`）、compressor state、indexer KV /
state。`DeepseekV4CacheLayout`（`:55`）/ `DeepseekV4CacheMetadata`（`:495`）描述这些
组的块表；`bind_paged_cache_scheduler` 有 override 做诊断。`clear_kv_buffers`
（`base.py:91`）里列的 `swa_kv_buffer` / `compressed_kv_buffer` /
`compressor_state_buffer` / `indexer_kv_buffer` / `indexer_state_buffer` 就是它的
buffer 名。

## 4. LCM two-level 分配（flat-state 模型）

hybrid / flat-state 模型（Qwen3.5 GDN、Kimi-K3 KDA、Inkling ShortConv）的 KV-cache
走统一的 **LCM two-level** 体系（PR #804）。它把异构的
history KV + linear-attention state 打包进**一块 budget 大小的 arena**，几何由单一
`cache_budget_bytes` 精确推导。（旧的 flat-slab 体系 `flat_hybrid.py` /
`flat_state_slabs.py` / `hybrid_cache_plan.py` / `flat_memory_plan.py` 已删除。）

> 📄 **本节是概览；完整展开见根目录 [`lcm.md`](lcm.md)**——独立成文，详解 LCM 名字
> 由来（parent 字节按所有 group 的 packing 因子求**最小公倍数（Least Common
> Multiple）**对齐，`lcm_memory_plan.py:401`）、几何层 / recipe / setup / pool 全链路、
> `exact_page_stride` 两类 kernel、端到端数值示例与文件索引。

### 4.1 核心概念

- 一个 **LCM block（parent）**是共享物理 arena 的唯一分配/调度单位，固定
  `_LOGICAL_BLOCK_TOKENS = 128` tokens（`lcm_setup.py:55`）。
- **two-level** = 一个 parent LCM block 里按每个 cache group 各自的
  `cache_blocks_per_lcm_block`（packing 因子）容纳固定整数个 **per-group 子 cache
  block（page）**。分配一个 parent 就同时为**所有** group（history KV pages、
  recurrent/conv state pages、draft history 等）预留好对应数量的 page。
- arena 保留 parent 0 作为永不调度的 **null LCM block**（`lcm_memory_plan.py:96`
  的 `arena_bytes` 含它，`num_lcm_blocks` 不含）。

### 4.2 几何计划 `LcmMemoryPlan`

`python/tokenspeed/runtime/configs/lcm_memory_plan.py`。纯整数几何，无 torch：

- 数据类：`LcmGroupLayout`（`:35`，`cache_blocks_per_lcm_block` + `page_count`）、
  `LcmFieldSpec`（`:42`）、`LcmPlaneLayout`（`:58`）、`LcmFieldLayout`（`:65`）、
  `LcmMemoryPlan`（`:80`，含 `group()` / `field()` / `plane()` 查询）。
- `plan_lcm_fields(...)`（`:217`）：规划一个 **plane-major** arena，field 在 cache
  group 间 overlay。容量可由 `budget_bytes` 推导或由 `num_lcm_blocks` 固定（后者用于
  colocated draft cache 保持 parent 几何一致）。内部：
  - `_solve_packing`（`:121`）：从 field 精确字节比推每组 packing。
  - `_check_exact_page_strides`（`:193`）：校验 kernel 隐式 payload-sized stride 与
    plane 给的 stride 一致（否则会读错行，fail loud）。
  - 每组 `page_count = 1 + num_lcm_blocks * count`（`:440`，含 dummy 页）。
  - 校验 padding fraction ≤ `max_padding_fraction`（`:425`）、kernel page id 不超
    `_MAX_KERNEL_PAGE_ID`（`:431`）。

### 4.3 布局 recipe 与 pool 构造

- **layout recipes**（`python/tokenspeed/runtime/configs/lcm_layouts.py`）：
  `mla_history_lcm_fields`（`:28`）/ `draft_history_lcm_fields`（`:55`）/
  `qwen_gdn_lcm_fields`（`:154`）/ `inkling_lcm_fields`（`:224`）/
  `kimi_k3_lcm_fields`（`:357`）。
- **setup 编排**（`python/tokenspeed/runtime/layers/attention/lcm_setup.py`）：
  `prepare_lcm_setup(...)`（`:578`）按 family 分派到 `_prepare_kimi_k3`（`:244`）或
  `_prepare_mha`（`:381`，覆盖 qwen_gdn / inkling），从单个 `cache_budget_bytes`
  同时给 target + draft arena 定尺寸；`create_lcm_pool(...)`（`:613`）按 config 类型
  造 `LcmMHATokenToKVPool*`（MHAConfig）或 `LcmMLATokenToKVPool`（MLAConfig）。
  `LcmPoolSpec`（`:60`）/ `LcmSetup`（`:83`）是产物。
- **物理存储 `LcmCachePool`**（`python/tokenspeed/runtime/layers/attention/kv_cache/lcm.py:31`）：
  持有单块扁平 `uint8` backing `torch.zeros(plan.arena_bytes, ...)`（`:36`），
  `field(field_id, dtype)`（`:39`）按 plan 用 `as_strided` 发 strided view，
  `zero_pages(page_ids_by_group)`（`:71`）用 `zero_byte_segments` 清零指定组的页，
  `pd_contract(...)`（`:80`）给 flat-KV PD 构造 contract。

### 4.4 LCM 计算接口

- `LcmMHATokenToKVPool`（`kv_cache/lcm_mha.py:44`）继承 `MHATokenToKVPool`：
  `_create_lcm_buffers`（`:141`）把 history 层的 `k_buffer/v_buffer` 绑到
  `lcm_pool.field(...)` 的 view，state 层绑 `(conv, ssm)` 到
  `_state_buffers_by_layer`；置 `flat_kv_requires_page_zeroing=True`（`:67`）；
  `zero_new_pages(new_page_ids)`（`:244`）供 scheduler 清零新分配的组页。
  MXFP8 变体 `LcmMHATokenToKVPoolMXFP8`（`:258`）再绑 scale field。
- `LcmMLATokenToKVPool`（`kv_cache/lcm_mla.py:40`）继承 `MLATokenToKVPool`：latent
  KV 层绑 `layer.{id}.latent_kv` field（`:160`），KDA state 层绑
  `(conv_state, recurrent_state)`（`:150`）——这就是 Kimi-K3 的布局。
- 两者都在构造时 `publish_paged_cache_groups(..., cache_blocks_per_lcm_block=...)`
  上报 group 与每组 `page_count`（`lcm_mha.py:102` / `lcm_mla.py:77`），PD 开启时给
  state 组打 `latest_snapshot`、history 组打 `full_suffix` transfer policy。
- 哪些模型走 LCM 由 `registry.py:1027`-`1035` 的 `lcm_family` 决定
  （`use_lcm_gdn` / `use_lcm_k3` / `use_lcm_inkling`），入口在
  `create_attn_components`（`registry.py:955`），LCM setup 在 `:1218`。

### 4.5 `cache_storage` 报告

`create_attn_components` 返回 8 元组的第 8 个 `cache_storage`
（`registry.py:1624`，由 `_cache_storage_report(...)` 在 `:195` 构造），只有跑了 LCM
setup 时非 `None`：描述**实际**分配字节 / token 容量 / LCM 几何
（`configured_cache_bytes` / `allocated_cache_bytes` / `physical_token_capacity` /
`geometry` 等），校验分配 ≤ budget。event_loop 挂到 `self.cache_storage`
（`event_loop.py:211`），随 scheduler ready payload 上报。

## 5. Prefix Caching / Radix 复用 + 驱逐

`python/tokenspeed/runtime/cache/prefix_cache.py`。这是 L1（GPU）上的 radix prefix
tree，用来跨请求复用相同前缀的 KV，避免重算。（运行时的页/驱逐调度很大程度由 C++
`tokenspeed_scheduler` 驱动，Python `PrefixCache` 是参考/radix 实现。）

- **抽象契约**（`cache/base_prefix_cache.py`）：`MatchResult` NamedTuple（`:30`）
  携带 `device_indices` / `last_device_node` / `device_prefix_length` /
  `last_host_node` / `host_hit_length`（后两者只在开 KVStore/L2 时非平凡）；
  `BasePrefixCache` ABC（`:52`）声明 `match_prefix` / `insert` / `evict` /
  `inc_lock_ref` / `dec_lock_ref`。
- **radix 树**（`prefix_cache.py`）：
  - `TreeNode`（`:92`）：`children` 是 defaultdict，含 `last_access_time` /
    `creation_time`，KVStore 字段 `host_ref_counter` / `host_value` /
    `hash_value`；`__lt__` 按 access time 排序。
  - `PrefixCache`（`:137`）持有 `req_to_token_pool`（`:79`）和
    `token_to_kv_pool_allocator`（`KVAllocator`，`:80`）。
  - **粒度是页，不是 token**：`match_prefix`（`:193`）把 key 按 `page_size` 切成
    页-tuple，每个 radix key 元素是一个页-tuple。
  - **命中**：`_match_prefix_helper`（`prefix_cache.py` 内）遍历 children，部分页
    匹配时 `_split_node` 分裂。
  - **驱逐**：`evict(num_tokens, evict_callback)`（`:381`）在
    `evictable_leaves` 上按 `eviction_strategy.get_priority` 建堆，弹出 victim，
    经 `append_to_later_free` 释放，再级联到变空的父节点。
  - **锁引用**：`inc_lock_ref`（`:419`）/ `dec_lock_ref`（`:434`）在
    `evictable_size_` / `protected_size_` 间移动，维护 `evictable_leaves`——正在
    forward 的请求的页被 protect，不会被驱逐。
  - **插入**：`cache_finished_req`（`:245`）/ `cache_unfinished_req`（`:313`）插页-
    tuple、重匹配、经 `free_with_diff`（`allocator.py:138`）释放重复设备页。
- **驱逐策略**（`cache/evict_policy.py`，策略模式）：`EvictionStrategy` ABC（`:37`），
  具体有 `LRUStrategy`（`:43`）/ `LFUStrategy`（`:48`）/ `FIFOStrategy`（`:53`）/
  `MRUStrategy`（`:58`）/ `FILOStrategy`（`:63`）/ `PriorityStrategy`（`:68`，priority
  + LRU tiebreak）。由字符串选，**默认 `"lru"`**。

## 6. 分层内存：GPU（L1）→ host（L2）→ 外部存储（L3）

非-flat 构建下，`MemoryExecutor` 是分层内存的顶层协调者，把冷 KV 从 GPU 换出到
host DRAM、再到外部 KV store，并在命中时预取回来。

### 6.1 三个层次与 owning class

- **L1 GPU**：radix prefix tree（第 5 节 `PrefixCache`，`prefix_cache.py:137`）。
- **L2 host DRAM**：pinned CPU 镜像。`HostExecutor`
  （`cache/executor/host_executor.py:109`）驱动 copy，针对 `HostKVCache` pool
  （`cache/kv_cache_host.py:128`）。
- **L3 外部存储**：`StorageExecutor`（`cache/executor/storage_executor.py:106`）over
  可插拔 backend（Mooncake）。
- **`MemoryExecutor`**（`cache/executor/memory_executor.py:114`）是协调者，**同时持有**
  `self.host_exec`（`:317`/`:319`）和 `self.storage_exec`（`:331`）。

### 6.2 `MemoryExecutor`

`cache/executor/memory_executor.py`：

- `MemoryExecutorConfig`（`:51`）：`host_ratio` / `host_size_gb` / `io_backend` /
  `host_layout` / `storage_backend` + mamba-L2 knob。
- `__init__`（`:114`）：拆 hybrid pool、自动 cap host size 到 cgroup budget、建
  DSA/MHA/MLA host pool、可选 draft + mamba host pool，再构造 `HostExecutor` 和
  `StorageExecutor`。
- `submit_plan(plan)`（`:361`）/ `submit(op)`（`:371`）：`WriteBackOp` / `LoadBackOp`
  → host_exec，`PrefetchOp` → `storage_exec.submit_prefetch`，`BackUpOp` →
  `submit_backup`。
- `poll_results()`（`:465`）：drain 两个 executor 的完成事件。
- `query_l3_pages(hashes)`（`:491`）：委托 storage 查 L3 命中页数。

### 6.3 L2 host 侧

- `HostExecutor`（`host_executor.py:109`）：独立 `write_stream` / `load_stream`，
  `enqueue_writeback` / `enqueue_loadback` 把页 id 转 token 索引、建 `TransferUnit`，
  per-layer copy 由 CUDA event 和 `LayerDoneCounter` 生产者/消费者 fence 门控。
- `HostKVCache` ABC（`kv_cache_host.py:128`）：按「显式 token 数 > GB > ratio」定 host
  池大小（`:146`），cgroup-aware 校验可用内存（`get_available_host_memory_bytes`,
  `:106`）；子类 `MHATokenToKVPoolHost`（`:282`）/ `MLATokenToKVPoolHost`（`:599`）/
  `DSATokenToKVPoolHost`（`:851`）实现 `load_to_device_per_layer` /
  `backup_from_device_all_layer`，支持 `layer_first` / `page_first` / `page_head` 布局。
- `MambaPoolHost`（`cache/mamba_cache_host.py:53`）：conv/ssm state 的 pinned host
  镜像（mamba L2），slot-based alloc。
- `LayerDoneCounter`（`cache/kvstore_controller.py:51`）：3-slot 生产者/消费者
  CUDA-event fence（`update_producer` / `wait_until`），layerwise load 用它对齐层。

### 6.4 L3 外部存储侧

- `StorageExecutor`（`storage_executor.py:106`）：线程池 + 单个 **aggregator 线程**
  （`_aggregator_loop`，`:339`），持一个专用 Gloo TP 子组、做 `ReduceOp.MIN`
  all_reduce 让各 rank 对命中/预取数达成一致；`_run_prefetch`（`:243`）/ `_run_backup`
  （`:304`）按 `op.rolling_page_hashes` 批量调 backend 的 `batch_get_v1` /
  `batch_set_v1`；`query_exists`（`:453`）数连续命中的 hash。
- `KVStoreStorage` ABC（`cache/kvstore_storage.py:49`）+ `KVStoreStorageConfig`
  （`:34`）；`batch_exists` 返回连续命中数。
- backend（`cache/storage/`）：`StorageBackendFactory`（`backend_factory.py:35`）按名
  懒加载，注册 `"mooncake"`；`MooncakeStore`（`storage/mooncake_store/mooncake_store.py:177`，
  继承 `KVStoreStorage`）包 `MooncakeDistributedStore`，用零拷贝
  `batch_get_into` / `batch_put_from`，key 后缀 `_{rank}_k/_v`（MHA）或 `_k`（MLA），
  `batch_set_v1` 跳过已存在 key。

### 6.5 flat 构建的精简版 `FlatMemoryExecutor`

`cache/executor/flat_memory_executor.py:93`：flat（M15）构建的 byte-blind 替代，
**无 host pool / 无 L3 / 无 mamba**。`FlatHostMirror`（`cache/flat_host_mirror.py:69`）
给每个不同的设备 tensor 一份 pinned 镜像；`FlatMemoryExecutor` 会 ACK loadback
（`emits_loadback_acks=True`，`:105`），`query_l3_pages` 恒返回 0。

### 6.6 L3 命中如何接入请求 admission（全在 `engine/event_loop.py`）

- 构造：`MemoryExecutorConfig`（`event_loop.py:318`）后分三路——flat 构建 →
  `FlatMemoryExecutor`（`:351`）；pool 不支持分层 → `self.memory_executor = None`
  （`:363`）；否则 `MemoryExecutor`（`:366`）。`_loadback_acks_expected` 从
  `emits_loadback_acks` 取（`:378`）。`HostExecutor` / `StorageExecutor` 只经
  `MemoryExecutor` 间接构造。
- L3 命中 → admission（`:1338` 一带）：`memory_executor` 非 None 时，
  `hashes = calc_l3_query_hashes(scheduler, spec.tokens)`（`:1339`，helper 在
  `:113`）；若 `len(hashes) > self.prefetch_threshold`（`:1340`，阈值来自
  `scheduler_cfg.prefetch_threshold`，硬编码 `4` 在 `:469`），
  `hit_pages = self.memory_executor.query_l3_pages(hashes)`（`:1341`），写
  `spec.rolling_hashes`（`:1349`）/ `spec.storage_hit_pages`（`:1350`）喂给 C++
  scheduler 规划预取。
- 提交/回收：`_submit_cache_ops`（`:1069`）调 `memory_executor.submit_plan`
  并计 inflight，`_setup_layerwise_loadback` 连生产/消费 fence；
  `_commit_cache_results`（`:770`）poll `memory_executor.poll_results()`、转 payload、
  rank 同步后 `scheduler.advance(...)`。这两者在调度循环里分别于 `:1740`/`:1897`
  提交、`:1730`/`:1882` 回收（见第 8 节）。

## 7. Flat cache page 清零（group-aware）

flat KV pool 把 recurrent state 和 KV 混叠在同一 arena，复用页时必须清零 tail，否则
上一个请求的 state「毒化」新请求。调度循环里：

- `_flat_page_ids_to_zero(execution_plan)`（`event_loop.py:1691`）：优先取
  `execution_plan.flat_pages_to_zero`（per-group child pages，`:1698`），旧单层
  scheduler 回退 `flat_page_ids_to_zero`（全局 page-id 向量，`:1701`）。
- `ModelExecutor.zero_flat_cache_pages(pages)`（`model_executor.py:1499`）：pages 是
  Mapping 且 pool 有 `zero_new_pages` 就调它（LCM 走这条，`lcm_mha.py:244`）；否则调
  `zero_pages`（page-id 列表）；**只有声明 `flat_kv_requires_page_zeroing` 的 pool
  缺 sanitizer 时才 fail loud**（`model_executor.py:1528` 一带）。返回一个
  `flat_cache_zero_event`（CUDA event），供 PD decode 侧 `_dispatch_forward` 在发布
  KV destination manifest 前 `.synchronize()` 等它（`event_loop.py:951`-`956`）。
- 调用点：`event_loop.py:1737`（普通循环）/ `:1894`（overlap 循环），event 随
  forward context 下传（`:1776` / `:1977`）。

## 8. 调度循环里的 KV 相关步骤

`event_loop()`（普通，`event_loop.py:1720` 一带）与 `event_loop_overlap()`
（`:1853` 一带）每轮涉及 KV 的关键步骤（详见 `inference-flow.md` 第 6 节）：

1. `_process_new_requests()`：收新请求，可选查 L3 命中页写入 spec（第 6.6 节），
   最终 `scheduler.submit_requests(...)`（`:763`/`:1276`）。
2. `_commit_cache_results()`（`:770`）：提交上一轮异步 L2/L3 cache op 结果。
3. `execution_plan = scheduler.next_execution_plan()`（`:1734`/`:1890`）：C++ 分配块、
   产出计划。
4. `_publish_scheduler_kv_events()`（`:818`）：发布 scheduler KV 事件。
5. `_handle_flat_oom_terminals(execution_plan)`（`:1432`，flat）：把永远塞不进 flat
   pool 的请求终止并报 abort。
6. `zero_flat_cache_pages(...)`（第 7 节）：清零本轮新分配的 flat page。
7. `_submit_cache_ops(execution_plan)`（`:1069`）：提交 L2/L3 load/write/prefetch。
8. `_dispatch_forward(...)`（`:886`）：执行 forward（PD 分支见第 9 节），过程中
   `update_block_table` 刷新 `req_to_page`（第 2.5 节）。
9. `advance_forward(self.scheduler, request_changes)`（`:1665` 等）：推进 C++
   scheduler FSM。

## 9. PD 分离下的 KV 跨节点传输

当 `disaggregation_mode != "null"`，prefill 节点算完 KV 后要发给 decode 节点。传输
基于 **Mooncake** RDMA/GPUDirect（`TransferBackend` 只有 `MOONCAKE` /
`MOONCAKE_ASYNC`，无 NCCL/NIXL——NCCL 仅用于 EPD embedding 行分片重组）。

### 9.1 cache-pool 抽象（`cache/transfer/`）

这是 host↔device writeback/loadback 的统一 pool 协议（也被网络传输复用底层设备
buffer）：

- `types.py`：`CacheKind`（`:29`，`KV` / `MAMBA`）、`Location`（`:34`，`DEVICE` /
  `HOST` / `STORAGE`）、`TransferUnit`（`:41`，一次 copy op：kind / src·dst_loc /
  src·dst_indices / op_id / is_retract）、`TransferBatch`（`:58`）。
- `pool.py`：`CachePool` Protocol（`:30`），声明 `writeback` / `loadback` /
  `copy_layer` / host-slot 分配 `alloc_host`/`free_host` 等。
- `kv_pool.py`：`KVCachePool`（`:29`）包一个 GPU `device_pool` + host offload
  `host_pool`（+ 可选 draft/MTP pool），建并注册 `LayerDoneCounter` 做层粒度进度。
- `mamba_pool.py`：`MambaCachePool`（`:33`）包 `SimpleMambaPool`（device）+
  `MambaPoolHost`；`page_size()==1`；实现 `copy_layer`（KV pool 不实现）。

### 9.2 event_loop 里的传输编排

- 创建：`disaggregation_mode != "null"` 时建 `kv_args` + `KVManagerArgs`，调
  **`create_kv_transfer(...)`**（`event_loop.py:611`）存 `self.kv_transfer`；非-PD
  节点为 `None`（`:648`）。`_setup_pd_layerwise_transfer`（`:652`）给 layerwise
  streaming 注册 step counter（prefill）。`create_kv_transfer`（`pd/factory.py`）按
  mode 造 `DisaggPrefillExecutor` / `DisaggDecodeExecutor`。
- **`_dispatch_forward` 的四条 path**（`event_loop.py:886`，docstring 列在 `:902`）：
  - **Path 1**（非 PD）：普通 forward。
  - **Path 2**（decode 节点 + EXTEND）：新请求等远端 KV——`reset_valid_cache_length`
    seed 远端算好的 prompt 长度，flat-KV 先
    `flat_cache_zero_event.synchronize()`（`:951`-`956`）确保页清零后再发布
    destination manifest，然后 `kv_transfer.execute(op)` 触发 **RDMA receive**，返回
    `(None, None)`。
  - **Path 3**（prefill 节点，无 EXTEND）：`kv_transfer.execute(op)` **发 KV** 给
    decode，返回 `(None, None)`。
  - **Path 3b**（decode 节点普通 decode forward）。
  - **Path 4**（prefill 节点 + EXTEND）：跑 prefill forward，`prepare_prefill(op)`
    做 layerwise streaming，返回 `on_first_token = kv_transfer.store_prefill_token`
    （`:1006`）。
- **first-token hand-off**：sampling 后 `generation_output_processor.py` 调
  `on_first_token(rid, ...)`，`store_prefill_token` 记录 first token（+ MTP spec
  candidates），经 Mooncake 控制面 ZMQ 状态消息带给 decode 侧。
- **register/abort**：admit 时 decode 设 `state.computed_length = input_length` 并
  `kv_transfer.register(rid, bootstrap)`（`event_loop.py:1335` 一带）开
  `MooncakeKVSender` / `MooncakeKVReceiver`。
- **事件泵**：每轮 `kv_transfer.generate_events()`（`:1788` / `:1990`）；
  `_process_kv_transfer_events`（`:1468`）处理 `SucceededEvent` /
  `RemotePrefillDoneEvent`（decode 侧写远端 spec candidate）/ `FailedEvent`。

### 9.3 交换的 manifest

遗留路径交换页索引向量 `kv_indices` + `aux_index` + `decode_prefix_len` + 可选
`mamba_indices` + `bootstrap_token`/`spec_candidate_ids`，以
`bootstrap_room`/`host`/`port` 为键。**FlatKV** 用结构化、带版本的
`FlatKVPDPageManifest`（`pd/flatkv.py`）：`groups`（per-group `page_ids`）+
`prefix_len` + `prompt_len` + `version`，收发双方经 `validate_flatkv_manifest_pair`
交叉校验。LCM pool 的 contract 由 `LcmCachePool.pd_contract(...)`（`lcm.py:80`）产出。

## 关键数据结构

| 数据结构 | 位置 | 作用 |
| --- | --- | --- |
| `BaseTokenToKVPool` | `layers/attention/kv_cache/base.py:37` | 所有物理 KV pool 基类；`supports_hierarchical_kv_cache` / `flat_kv_requires_page_zeroing` 类属性。 |
| `PagedCacheGroupSpec` | `configs/paged_cache_spec.py:28` | 一个 cache group 的布局/保留策略（retention / rows_per_page / family / packing）。 |
| `MHATokenToKVPool` | `layers/attention/kv_cache/mha.py:53` | 标准 MHA/GQA KV pool（含 hybrid slab、异构 head、MXFP8 子类）。 |
| `MLATokenToKVPool` | `layers/attention/kv_cache/mla.py:50` | MLA 压缩 latent KV pool（默认 / per_token_head 量化）。 |
| `DSATokenToKVPool` | `layers/attention/kv_cache/dsa.py:37` | sparse MLA（GLM-5），额外存 lightning-indexer index_k。 |
| `DeepseekV4TokenToKVPool` | `layers/attention/kv_cache/deepseek_v4.py:755` | DeepSeek-V4 多 buffer pool（SWA / 压缩 KV / compressor / indexer）。 |
| `LcmCachePool` | `layers/attention/kv_cache/lcm.py:31` | 单块 LCM arena backing，按 field 发 strided view 的 flat-state pool 基类。 |
| `LcmMemoryPlan` | `configs/lcm_memory_plan.py:80` | two-level LCM arena 的纯整数几何（parent ↔ per-group child pages 打包）。 |
| `LcmMHATokenToKVPool` / `LcmMLATokenToKVPool` | `kv_cache/lcm_mha.py:44` / `lcm_mla.py:40` | LCM arena 上的 MHA/MLA 计算接口（history + state 共享 arena）。 |
| `KVAllocator` | `cache/allocator.py:30` | radix 路径的纯元数据页分配器（`free_slots` / `req_to_page`）。 |
| `ReqToTokenPool` | `cache/req_to_token_pool.py:48` | 请求 → 物理 token 位置表 `[size, max_context_len]`。 |
| `PrefixCache` / `TreeNode` | `cache/prefix_cache.py:137` / `:92` | GPU 端 radix prefix 复用树（页粒度）+ 驱逐。 |
| `EvictionStrategy` | `cache/evict_policy.py:37` | 驱逐策略（LRU 默认 / LFU / FIFO / MRU / FILO / Priority）。 |
| `MemoryExecutor` | `cache/executor/memory_executor.py:114` | 分层内存顶层协调（持 host_exec + storage_exec）。 |
| `HostExecutor` / `HostKVCache` | `cache/executor/host_executor.py:109` / `cache/kv_cache_host.py:128` | L2 host DRAM 传输 + host pool。 |
| `StorageExecutor` / `MooncakeStore` | `cache/executor/storage_executor.py:106` / `cache/storage/mooncake_store/mooncake_store.py:177` | L3 外部存储异步预取/回写 + Mooncake backend。 |
| `FlatMemoryExecutor` / `FlatHostMirror` | `cache/executor/flat_memory_executor.py:93` / `cache/flat_host_mirror.py:69` | flat 构建的精简分层（只 host mirror，无 L3）。 |
| `LayerDoneCounter` | `cache/kvstore_controller.py:51` | layerwise KV load 的 3-slot CUDA-event fence。 |
| `KVCachePool` / `MambaCachePool` | `cache/transfer/kv_pool.py:29` / `mamba_pool.py:33` | PD/offload 的 cache-pool 抽象（包 device+host buffer）。 |
| `TransferUnit` / `TransferBatch` | `cache/transfer/types.py:41` / `:58` | 一次 KV/state copy 操作及其批。 |

## 读代码建议

只想快速掌握 KV 主线，按此顺序读：

1. `python/tokenspeed/runtime/configs/paged_cache_spec.py`（cache group / page /
   block 的定义与发布）
2. `python/tokenspeed/runtime/layers/attention/kv_cache/base.py`（pool 基类契约）
3. `python/tokenspeed/runtime/layers/attention/kv_cache/mha.py` 和 `mla.py`（两种
   最基本的物理布局）
4. `python/tokenspeed/runtime/engine/event_loop.py:395`-`507`（Python 如何把几何交给
   C++ scheduler 并绑定）
5. `python/tokenspeed/runtime/execution/model_executor.py:1489`（每步块表更新）和
   `execution/cache_loc_kernel.py`（物理写位置计算）
6. `python/tokenspeed/runtime/cache/prefix_cache.py` + `cache/evict_policy.py`
   （prefix 复用与驱逐）

想深入特定子系统：

- **LCM two-level（flat-state）**：`layers/attention/lcm_setup.py`、
  `configs/lcm_memory_plan.py`、`configs/lcm_layouts.py`、
  `kv_cache/lcm.py` / `lcm_mha.py` / `lcm_mla.py`。
- **分层内存（L2/L3）**：`cache/executor/`（`memory_executor.py` /
  `host_executor.py` / `storage_executor.py` / `flat_memory_executor.py`）、
  `cache/kv_cache_host.py`、`cache/mamba_cache_host.py`、`cache/kvstore_controller.py`、
  `cache/storage/`。
- **PD KV 传输**：`cache/transfer/`（`pool.py` / `kv_pool.py` / `mamba_pool.py` /
  `types.py`）、`runtime/pd/`（`factory.py` / `flatkv.py` / prefill·decode executor）、
  `engine/event_loop.py:886`（`_dispatch_forward` 四条 path）。
- **特殊模型 pool**：`kv_cache/dsa.py`（GLM-5 sparse MLA）、
  `kv_cache/deepseek_v4.py`（DeepSeek-V4 多 buffer）、`kv_cache/msa.py`（MiniMax M3）。
