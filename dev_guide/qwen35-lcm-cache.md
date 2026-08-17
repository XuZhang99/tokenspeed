# Qwen3.5 混合注意力的 LCM Cache 管理

本文以 Qwen3.5 的 full attention + Gated DeltaNet（GDN）linear attention 为例，
说明 TokenSpeed 如何用一套 LCM cache 同时管理逐 token 的 K/V history 和 recurrent
state checkpoint。通用对象与术语见 [KV Cache 管理机制](kvcache-management.md) 和
[LCM 两级 Cache 分配机制](lcm.md)。

这里的 LCM 指代码中的等宽物理 parent/packing 机制，不是把 full attention 与 linear
attention 的 token 窗口直接做一次 `math.lcm()`；最小公倍数运算只用于求整数 packing、
stride 和 alignment 等布局约束。

本文描述的是当前实现，不是旧的独立 Mamba request-state pool：Qwen3.5 的 GDN
`conv/ssm` state 已经进入统一 `CacheArena`，和 full-attention K/V 共享一套物理
parent allocator、prefix cache 和 block-table 生命周期。

## 1. 结论先行

Qwen3.5 的 cache 管理可以概括为：

```text
Qwen3.5 layer pattern
  linear-0, linear-1, linear-2, full, ...
                  │
                  ▼ QwenGDNRecipe
  state-0 group + state-1 group + state-2 group + full group
                  │
                  ▼ pack()
  不同大小的 state block / KV block overlay 成等宽 LCM parent
                  │
                  ▼ bind(N)
  一个 CacheArena + N 个可用 parent + 1 个 null parent
                  │
                  ▼ C++ CacheCoordinator
  每个请求、每个 group 各有 block table；所有 group 竞争同一个 BlockPool
                  │
          ┌───────┴────────┐
          ▼                ▼
  full-attention backend   GDN backend
  消费 full group table    消费三个 state group table
  读写 K/V token rows      按 state_in/state_out 双索引读写 checkpoint
```

最重要的四个事实是：

1. 一个 full group 的 child block 保存所有 full-attention 层在 128 token 内的 K/V；
   一个 state group 的 child block 保存该位置组内所有 GDN 层的一份 conv/ssm checkpoint。
2. 多个 group 的字段可以 overlay 同一套 parent 字节几何，但同一 parent 有 live child
   时只能绑定一个 group，不能同时放 full K/V 和 state。
3. scheduler 输出的是 group-scoped child block id。`packing > 1` 时它不是 LCM parent
   id，forward 路径不再自行计算 packing。
4. full history 永不因窗口过期；GDN 只保留计算所需的当前 state/边界 snapshot，旧
   checkpoint 可进入 prefix cache，也可被淘汰。

## 2. 从模型层到 cache group

### 2.1 layer type 归一化

Qwen checkpoint 把 full attention 写成 `"attention"`。配置层在
[`qwen3_5_text_base_config.py`](../python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py)
中将其转换成 cache 侧统一标签：

```text
checkpoint "attention"  -> "full_attention"
checkpoint "linear_attention" -> "linear_attention"
```

典型的 `full_attention_interval = 4` 产生：

```text
layer type: linear, linear, linear, full, linear, linear, linear, full, ...
```

具体模型的 interval 和层数可以不同；cache recipe 读取最终的 `layer_types`，不把
“3:1”写死在 planner 中。

### 2.2 为什么 linear attention 要拆成多个 state group

[`QwenGDNRecipe`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py)
调用 `split_recurrent_state_groups()`，按一段 recurrent run 中的位置拆组。对 3:1
模式，逐层 group id 为：

```text
layer:     0                    1                    2                    3
type:      linear               linear               linear               full
group:     linear_attention_0   linear_attention_1   linear_attention_2   full_attention

layer:     4                    5                    6                    7
type:      linear               linear               linear               full
group:     linear_attention_0   linear_attention_1   linear_attention_2   full_attention
```

因此所有 full-attention 层共享一个 history group；同一 recurrent 位置在不同重复
block 中的层共享一个 state group。每个 state group 有独立 block table、prefix
checkpoint 和物理 placement，backend 再通过 `group_id_for_layer(layer_id)` 找到本层
应使用的 group。

这样拆组也让同一个 occurrence 的物理 plane 能按以下方式 overlay：

```text
unit.i.a: full layer i 的 K
          或 state-0/1/2 在第 i 次 occurrence 的 SSM state

unit.i.b: full layer i 的 V
          或 state-0/1/2 在第 i 次 occurrence 的 Conv state
```

这里的“或”很关键：allocator 通过 parent 的 group binding 保证同一时刻只有一种解释
有效。

## 3. Qwen3.5 声明的逻辑与物理字段

Qwen recipe 固定：

```text
prefix_granularity P = 128 tokens
```

这与用户配置的 attention kernel page size 是不同概念。当前 Qwen GDN group 的
`block_granularity` 也为 128，但 kernel page size 仍由 backend 单独决定，不能从 P
反推。

每个 full-attention 层声明：

```text
K shape = [128, local_kv_heads, head_dim]
V shape = [128, local_kv_heads, head_dim]
dtype   = configured KV cache storage dtype
```

每个 GDN 层声明：

```text
SSM  shape = [local_value_heads, value_head_dim, key_head_dim]
Conv shape = [local_conv_channels, conv_kernel_size - 1]
SSM  dtype = TOKENSPEED_MAMBA_SSM_DTYPE 对应的 dtype
Conv dtype = bfloat16
```

SSM field 要求 exact page stride；Conv field 使用 flexible stride，可以占用 V plane
剩余的 slack。shape 已包含 attention TP 切分，所以不同 TP、KV dtype、head 配置会得到
不同 packing，不能给所有 Qwen3.5 变体写死同一个比例。

target 的 MXFP8 interleaved-scale KV layout 当前会被 Qwen recipe 拒绝；普通 BF16/FP8
KV storage 与 draft 自己声明的字段按 memory plan 处理。

## 4. LCM parent 如何求出

### 4.1 packing 的含义

对 group `g`：

```text
K_g = cache_blocks_per_lcm_block[g]
```

表示一个 LCM parent 绑定到 `g` 后，可以切成多少个 `g` 的 child CacheBlock。planner
根据共享 exact-stride plane 上的 payload 字节比例求最小整数比例，并同时满足 dtype、
stride、256-byte alignment 和 padding 限制。

简化地看，如果同一 `unit.i.a` plane 上：

```text
1 个 SSM block 的字节数 = 16 × 1 个 K block 的字节数
```

那么可以得到：

```text
full_attention packing = 16
state group packing     = 1
```

此时一个 parent 绑定 full group 时可放 16 个 128-token K/V child blocks；绑定任一
state group 时放 1 个 state checkpoint block。实际比例由
[`pack()`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py) 从当前模型
几何计算；`16:1` 是仓库 Qwen3.5 FP8 scheduler case 使用的典型示意，不是跨模型常量。

### 4.2 parent 是 overlay，不是静态分区

物理布局可以理解成：

```text
LCM parent j
├─ unit.0.a plane: 16 个 full child 的 occurrence-0 K
│                   或 1 个 state child 的 occurrence-0 SSM
├─ unit.0.b plane: 16 个 full child 的 occurrence-0 V
│                   或 1 个 state child 的 occurrence-0 Conv
├─ unit.1.a plane: 16 个 full child 的 occurrence-1 K
│                   或 1 个 state child 的 occurrence-1 SSM
├─ unit.1.b plane: 16 个 full child 的 occurrence-1 V
│                   或 1 个 state child 的 occurrence-1 Conv
└─ ...
```

但 allocator 状态实际是：

```text
parent = free
      或 bound(full_attention, occupancy[0..K_full-1])
      或 bound(linear_attention_0, occupancy[0..K_state0-1])
      或 bound(linear_attention_1, occupancy[0..K_state1-1])
      或 bound(linear_attention_2, occupancy[0..K_state2-1])
```

只要一个 sibling child 仍被 request table 或 prefix cache 引用，parent 就不能切换到
另一个 group。当前 group 再申请 block 时，`BlockPool` 会优先填充该 group 已部分占用
的 parent，再取一个 free parent，减少内部碎片。

### 4.3 parent 字节数

对每个 plane：

```text
plane_bytes = aligned_max(K_g × bytes_of_group_g_on_this_plane)
```

所有 plane 相加并对齐后得到：

```text
lcm_block_bytes = bytes of one byte-uniform parent
```

Conv 是 flexible-stride 字段，通常利用由 exact K/V 或 SSM 字段确定的 plane 宽度，
不会反向要求 kernel 假设 payload-contiguous stride。

## 5. 从显存预算绑定容量

设 cache 预算为 `B`，speculative verify workspace 为 `W`，每个 parent 为 `L` bytes，
则 Qwen recipe 的预算上限为：

```text
N_budget = floor((B - W) / L) - 1
```

减掉的 1 是 null parent。若配置了 token limit，还会按
`max_packing × P` 对 parent 数做上限裁剪。最后：

```text
plan = layout.bind(N)

arena_bytes = (N + 1) × L
group_page_count[g] = 1 + N × K_g
plan_token_slots = N × max(K_g) × P
```

`page_count` 中的 `+1` 是 group 的 null block 0；`N` 只统计可用于 admission 的
parent。

`plan_token_slots` 是 packing 最细 group 的行空间，不等于混合 group 同时满载时的
实际 admission。C++ scheduler 会按每组需求计算：

```text
parents_needed = sum_g ceil(child_blocks_needed[g] / K_g)
```

并用共享 parent occupancy、可淘汰 prefix 和 in-flight reservation 做最终 admission。
因此诊断 OOM/拒绝调度时，应看各 group child demand 和 parent binding，不能只看
`N × max_packing × P`。

## 6. Arena、page id 与字段 view

[`CacheArena`](../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py) 一次性分配
`uint8` buffer，并按 plan eager 创建所有 typed/as-strided view：

- full K/V 的 shape 以 P 开头，page 轴会折成连续 token-row 轴；
- conv/ssm shape 不以 P 开头，保留 `[page_count, *state_shape]` 的 block-major view；
- target 与 MTP draft 只是同一 arena 的 layer window，不重复分配 cache。

Scheduler 内部用：

```text
CacheBlockLocation(parent_id, slot_index)
```

对 group `g` 导出给 kernel/runtime 的 child block id 为：

```text
page_id = 1 + (parent_id - 1) × K_g + slot_index
```

例如 `K_full = 16` 时，parent 7 的 full child id 是 97..112；`K_state = 1` 时，
parent 7 对相应 state group 的 child id 是 7。parent 7 不会同时采用这两种解释。

同一个整数 id 也必须放在 group 作用域下理解：

```text
block_tables["full_attention"]
block_tables["linear_attention_0"]
block_tables["linear_attention_1"]
block_tables["linear_attention_2"]
```

字段的最终字节 offset 统一由 `CacheMemoryPlan.field_page_byte_offset()` 计算。forward、
清零、PD 和 Host L2 都不应复制 parent/slot 公式。

## 7. Scheduler 如何管理两类 cache

### 7.1 runtime contract

Arena 发布的 `CacheRuntimeContract` 包含：

```text
P、N、token_capacity
每个 group 的 retention/family/block_granularity
每个 group 的 page_count 和 packing
```

Python bridge 将其转换成 C++ `CacheGroupConfig`：

```text
full_attention       -> kind = Full,       family = History
linear_attention_*   -> kind = MambaState, family = State
```

snapshot state 在 bridge 处编码为
`rows_per_page=checkpoint_granularity, entry_stride_tokens=1`，C++ 统一读取
`block_granularity`，不接触 state tensor shape。

### 7.2 full-attention 生命周期

full group 是 prefix-closed：

- 每 128 token 一个逻辑 CacheBlock；
- request 的 table 按上下文增长；
- 完整 block 可以按 prefix hash 注册复用；
- full history 不因窗口滑动而过期；
- 内存紧张时，无 live owner 的 cached block 可以按 cache 淘汰策略回收。

### 7.3 GDN state 生命周期

state group 的一个 table slot 表示“每 128 token 保存的一份 recurrent checkpoint”，
不是 128 行 token state。GDN 计算只需要边界前的 state 和本次更新后的 state，因此
scheduler 使用 Mamba-state retention window，在跨 checkpoint 时保留 input snapshot
和 output working block；更老的 request-table block 会过期。

在恰好 128-token 边界生成的 state 可以注册成 prefix checkpoint。chunked prefill
budget 会向下对齐到所有 full-history state group checkpoint grain 的最小公倍数；对
当前 Qwen group 都是 128，因此不对齐的 chunk 不会被当成可复用 state 边界。

Prefix probe 会跨 group 收敛可恢复边界：full group 提供完整 history 命中，state
group 在相同边界提供恢复计算所需的 checkpoint。不能只命中 K/V 而忽略 state，反之
亦然。

## 8. Forward 时怎样读写

### 8.1 full attention

MHA backend 先丢弃 family=`state` 的 tables，只消费 full history table。每层按自己的
`group_id` 选择 table，计算 K/V 写入位置，再通过本层绑定的 K/V field view 读写。

由于 scheduler 已把 `(parent, slot)` 转成 child `page_id`，MHA forward 只做 group
block 到 kernel page 的必要转换；它不再处理 `cache_blocks_per_lcm_block`。

### 8.2 linear attention 的双索引

GDN backend 对每个 state group、每个 batch 只计算一次 `state_in/state_out`，然后各层
按 group id 复用结果。设：

```text
before = 本次 forward 前的 sequence length
after  = 本次 forward 后的 sequence length
P      = checkpoint_granularity = 128
```

则逻辑 table slot 为：

```text
in_slot  = max(floor((before - 1) / P), 0)
out_slot = floor((after - 1) / P)
```

随后从每个 state group 的 block table gather 出实际 child block id：

```text
state_in_blocks[group]
state_out_blocks[group]
```

- 新请求 `before == 0` 时，`state_in` 使用 null block 0，并显式 mask 成 zero state；
- 未跨 128-token 边界时允许 `state_in == state_out`，在同一 working block 内演进；
- 跨边界时 input snapshot 与 output block 必须不同；
- layer 通过 `group_id_for_layer()` 选 group，再通过 `get_component()` 取得该层 conv/ssm
  field view。

这也是为什么三个 recurrent 位置必须保持独立 group table：同一个 batch 上，它们的
物理 child id 可以完全不同，不能拿一张 state table 代替三张。

## 9. Prefix reuse、淘汰与清零

一个常见生命周期如下：

```text
1. ProbePrefix
   full + state groups 一起确定公共可恢复边界

2. Admit
   pin 命中 block；为缺失 suffix 分配 child block
   allocator 先填同 group 的 partial parent，再取 free/evictable parent

3. pages_to_zero
   scheduler 返回本轮新分配的 {group_id: child ids}

4. zero_new_blocks
   HybridMHATokenToKVPool 按 plan 清零这些 block 的全部 field payload

5. Forward
   MHA 写 K/V；GDN 用 state_in/state_out 更新 conv/ssm

6. CacheCompletedBlocks / ReclaimExpired
   注册对齐的 prefix/checkpoint；释放 GDN 已滑出的 request-table owner

7. Evict / Free
   最后一个 child owner 释放后 parent 才能解绑并交给另一个 group
```

Qwen hybrid pool 设置 `requires_page_zeroing=True`。这是正确性要求：一个刚从 full
group 解绑的 parent 可能残留 FP8/BF16 K/V；若直接按 FP32 SSM state 解释，可能得到
巨大值或 NaN。fresh block 必须在 forward 前按 group field range 清零。除此之外，
fresh sequence 的 initial recurrent state 还会在 backend 再做一次 zero mask。

## 10. MTP speculative decoding

Qwen MTP draft 层按“同一个大模型的 continuation layer”加入 target plan：

- draft 层都是 full attention，加入 `full_attention` group；
- target/draft 共用一个 memory plan、一个 arena、一个 runtime contract；
- draft K/V plane 会增大 full binding 的 parent 字节，并让 state binding 在 draft-only
  plane 上产生结构性 padding；Qwen recipe 的 padding bound 显式为此留出空间；
- target verify 不把每个 speculative state 都直接写进持久 state block，而是在
  graph-stable scratch 中计算，最终只把 accepted position 对应的 conv/ssm state commit
  到 scheduler 已分配的 destination block；
- 开启且 kernel 支持 ReplaySSM 时，verify scratch 只暂存 conv，SSM 在 commit 时 replay；
  否则 conv 和 SSM 都需要 staging workspace。

因此 speculative reject 不会污染持久 recurrent checkpoint，draft 也不会拥有另一套
与 target 脱节的 LCM page-id 空间。

## 11. 数值示例

假设某个 Qwen3.5 plan 得到：

```text
P = 128
N = 100 usable LCM parents
K_full = 16
K_state0 = K_state1 = K_state2 = 1
```

则：

| group | 可用 child blocks | 含 null 的 page_count | 单 child 逻辑跨度 |
| --- | ---: | ---: | ---: |
| `full_attention` | 1600 | 1601 | 128 tokens K/V rows |
| `linear_attention_0` | 100 | 101 | 128-token checkpoint |
| `linear_attention_1` | 100 | 101 | 128-token checkpoint |
| `linear_attention_2` | 100 | 101 | 128-token checkpoint |

若某次 admission 需要 20 个 full child blocks，并且三个 state group 各需要 1 个
child block，则无视已有 partial parent/命中时的 parent 下界为：

```text
ceil(20 / 16) + ceil(1 / 1) + ceil(1 / 1) + ceil(1 / 1)
= 2 + 1 + 1 + 1
= 5 parents
```

这说明“full group 有 1600 个 child id”不代表另外三个 state group 各自静态预留了
100 个 parent。它们只是各自拥有可寻址 id 空间，运行时动态竞争同一批 100 个物理
parent。

## 12. 常见误区

- **把 LCM parent 当成 token page。** Parent 是等宽物理字节单位；每组一个 child
  block 的 token 含义由 `block_granularity` 定义。
- **认为 group 同时占据 parent 的不同区域。** 字段 view 可以 overlay，但 live
  ownership 是 parent 级 group-exclusive。
- **把 block table entry 当 parent id。** `packing > 1` 时 entry 是 affine 展开的
  child id，必须保留 group 作用域。
- **让 state backend 使用 full table。** full backend 会 shed state tables；GDN 必须
  从 `state_in_blocks_by_group/state_out_blocks_by_group` 取值。
- **为 GDN 另建 request-indexed state pool。** 当前生产路径的 conv/ssm 已由 LCM
  arena 和 C++ scheduler 管理。
- **只按 `N × max_packing × P` 判断容量。** 混合模型 admission 要把每组 child
  demand 除以各自 packing 后相加，并考虑 parent binding/fragmentation。
- **复用 parent 时不清零。** group overlay 会让旧 dtype 字节污染 recurrent state。
- **把 16:1 写死。** packing 取决于模型 shape、TP 和 cache dtype，应读取 runtime
  contract 或启动日志中的 `groups={...}`。

## 13. 排障入口

启动时关注 cache profile 日志：

```text
Cache profile: parent_bytes=..., P=128, parents=..., token_capacity=...,
layers=... (draft ...), groups={...}
```

建议依次核对：

1. `CachePoolSpec.layer_group_ids` 是否符合模型 layer pattern；
2. `CacheMemoryPlan.groups` 中每组 packing/page_count 是否合理；
3. `CacheArena.runtime_contract` 是否完整发布 history + state groups；
4. scheduler `block_tables` 的 key 是否与 contract group id 完全一致；
5. fresh state child id 是否进入 `pages_to_zero`；
6. GDN layer 是否通过 `group_id_for_layer()` 选择了正确 state table；
7. chunked prefill 是否对齐 128-token checkpoint 边界；
8. MTP verify 是否只 commit accepted state。

主要源码入口：

- [`recipes/qwen35.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py)
- [`recipes/spec.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/spec.py)
- [`recipes/plan.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/plan.py)
- [`kv_cache/arena.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/arena.py)
- [`kv_cache/hybrid_mha.py`](../python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_mha.py)
- [`backends/hybrid_linear_attn.py`](../python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py)
- [`engine/scheduler_utils.py`](../python/tokenspeed/runtime/engine/scheduler_utils.py)
- [`BlockPool`](../tokenspeed-scheduler/csrc/cache/core/block_pool.h)
- [`GroupAllocator`](../tokenspeed-scheduler/csrc/cache/allocator/group_allocator.h)
- [`CacheCoordinator`](../tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.cpp)
