# PD 分离（Prefill / Decode Disaggregation）

当前 PD cache 协议直接传统一 `CacheMemoryPlan` 与 `CacheGroupSpec`，不再维护旧的
FlatKV/LcmCachePool 专用 schema。实现目前只支持 Mooncake backend。

## 总览

```text
Decode admission
  ├─ scheduler 分配 destination CacheBlock tables
  ├─ build CachePDBlockManifest
  └─ MooncakeKVReceiver 发起 bootstrap/pre-allocation
                     │ control: HTTP + ZMQ
                     ▼
Prefill
  ├─ 本地执行 prompt
  ├─ producer schedule 标记 field ready barrier
  ├─ CacheTransferPlanner 处理 TP 映射
  └─ Mooncake RDMA 写入 Decode arena
                     │
                     ▼
Decode
  ├─ 接收 bootstrap token / speculative candidates
  ├─ RemotePrefillDoneEvent 回灌 C++ FSM
  └─ 从已传输 cache 开始本地 decode
```

## 1. 目录

```text
python/tokenspeed/runtime/pd/
├── factory.py              # contract 与 executor 创建
├── cache_protocol.py       # wire contract / manifest / validation
├── transfer_plan.py        # 不同 TP 的 field fragment 规划
├── prefill_executor.py
├── decode_executor.py
├── base/
│   ├── bootstrap.py        # HTTP route 服务
│   ├── manager.py          # status/failure 共用状态
│   ├── mooncake_engine.py  # Mooncake TransferEngine 封装
│   └── status.py
└── mooncake/
    ├── conn.py
    ├── entities.py         # ZMQ/wire entities
    ├── prefill.py
    ├── decode.py
    ├── sender.py
    └── receiver.py
```

## 2. 启动与角色

`disaggregation_mode` 选择 `prefill`、`decode` 或非 PD。cache recipe 在 P/D 角色下为
每个 group 填 transfer policy：

- `full_suffix`：history 或 rolling/sliding state；
- `latest_snapshot`：full-history recurrent state，只传最终 snapshot。

`factory.create_kv_transfer()` 当前只接受 `TransferBackend.MOONCAKE`：

- Prefill → `DisaggPrefillExecutor`；
- Decode → `DisaggDecodeExecutor`。

模型能力还可能限制组合。例如当前 Inkling PD 是 target-only，DeepSeek V4 draft cache
也不支持传输；这些 gate 位于 attention registry，不能只看 transfer backend 判断。

## 3. Arena contract

`factory.get_kv_args()` 从 target pool 的合并 arena 生成一份注册：

1. `build_cache_transfer_schema(plan, model_config, draft_model_config)` 描述 TP partition；
2. `build_cache_fields_by_producer_step(plan, num_target_layers)` 描述每个 prefill barrier
   后哪些 field 已准备好；
3. `build_arena_cache_transfer_contract(arena, schema)` 返回 typed contract 与 owner
   base address。

target/draft continuation fields 已在同一个 plan/arena，不能重复注册 draft slab。

### 3.1 `CacheTransferContract`

[`cache_protocol.py`](../python/tokenspeed/runtime/pd/cache_protocol.py)：

```python
@dataclass(frozen=True, slots=True)
class CacheTransferContract:
    plan: CacheMemoryPlan
    group_specs: tuple[CacheGroupSpec, ...]
    transfer_schema: CacheTransferSchema
```

plan 已携带 field dtype、shape、group、plane、offset 与 stride；不再传
`field_dtypes` map。`build_cache_transfer_contract()` 还验证 backing 必须是：

- contiguous `uint8` owner；
- storage offset 0；
- `data_ptr == untyped_storage.data_ptr`；
- `nbytes == plan.arena_bytes`。

contract JSON wire 有 256 KiB 上限。反序列化后会重建 plan/spec/schema dataclass 并
再次验证 schema。

### 3.2 `CacheProducerSchedule`

它是 Prefill 本地信息，不上 wire：按 global layer 将 field 分到 producer step。
target 每层一个 barrier；draft continuation 的所有 field 合并到最后一个 barrier，
因为 draft physical layer 可能重复执行。

## 4. 请求 block manifest

`CachePDBlockManifest` 只传某个请求的逻辑选择：

```text
groups: (group_id, block_ids[])[]
prefix_len
prompt_len
```

manifest wire 上限 2 MiB。Decode 根据 scheduler 的 destination block tables 构建并
先完整校验一个 batch，再发布任何 receiver，避免前一行已等待、后一行验证失败造成
永久悬挂。

`build_cache_block_manifest()` 根据每组 transfer policy 选择 slot：

- `full_suffix`：从未命中 prefix 到 prompt 尾；sliding group 只保留下一 token 能
  attend 的 window tail；
- `latest_snapshot`：选择 prompt 尾对应的最终 state block。

Prefill 侧用自己的 source table 与 Decode manifest 位置建立
`CachePDLayerwiseBlockSelection`，layerwise chunk 只引用 destination manifest 的位置，
不发一份语义不完整的 partial manifest。

## 5. Peer layout 校验

传输前，Prefill/Decode contract 必须在逻辑语义上兼容：

- group id/order、retention、family、policy；
- prefix/granularity 与 field 集；
- dtype 和可映射的 shape/partition；
- plan 的 group/plane/field 自洽性。

两端 `num_lcm_blocks`、arena base address 和 rank-local shape 可以不同；它们是 peer
本地物理事实。传输 planner 在已验证 contract 上解析 source/destination offset，
不会把 peer-local base/stride 固化成第二套可信 ABI。

## 6. TP transfer planner

[`transfer_plan.py`](../python/tokenspeed/runtime/pd/transfer_plan.py) 的
`CacheTransferPlanner` 支持 Prefill TP 与 Decode TP 不同。

`CacheTransferSchema` 为需要切分的 field 指定 partition axis、global extent 与各 rank
分片。planner 对一个 Decode rank 计算：

```python
RankTransferPlan(
    fragments_by_prefill_rank={
        prefill_rank: (CacheTransferFragment, ...),
    }
)
```

fragment 描述 field-relative row copy：group/field、source/destination field offset、
row stride、bytes per row 与 rows per block。arena base、block id 对应的 offset 在执行
时从双方已验证 plan 解析。

当 TP 相等时使用同 rank、无额外 fragment 的 fast path；不分片 field 选择 replicated
source rank；分片 field 用 global interval intersection 生成若干 fragment。TP 大小
上限为 `MAX_CACHE_TP_SIZE=1024`。

## 7. Mooncake 控制面

### 7.1 HTTP route

Prefill manager 启动/注册 bootstrap 信息；Decode receiver 查询 Prefill 的 DP/TP 拓扑
和目标 engine rank。route key 包含 target DP group 与 TP 路由，避免请求落到不匹配
的注册。

### 7.2 ZMQ bootstrap

Decode 把 destination registration、manifest、peer contract、session/room 信息发给
Prefill。Prefill 校验后记录 `TransferInfo`，为每个 Decode destination 建立传输路线。

wire parser 对整数长度、ASCII、frame 数、contract/manifest 大小都有边界检查；任何
解析异常应把 room 标成 Failed，而不是杀死 bootstrap worker。

### 7.3 状态

`TransferPoll` 单调推进：

```text
Bootstrapping → Bootstrapped → WaitingForInput → Transferring → Success
       └──────────────────────────────────────────────────────→ Failed
```

`Failed` 粘滞。多 rank poll 通过 Gloo all-reduce 收敛；executor 只在状态 transition
时生成一次 scheduler event，并清理 terminal request state，避免重复 FailedEvent。

## 8. Prefill executor

`DisaggPrefillExecutor` 维护 per-request sender 与本地状态。主要职责：

1. bootstrap，等待 Decode 注册；
2. 消费 Prefill forward operation；
3. 在 producer barrier 到达时选择已 ready fields/block；
4. 按 planner 路线向一个或多个 Decode rank RDMA 写入；
5. 等待 bootstrap token/speculative candidates 准备；
6. 发送 success metadata；
7. 产生 `PD.SucceededEvent` 或 `PD.FailedEvent`。

同一 room 的多个 destination 必须对 prefix/prompt window 达成一致。请求结束或失败
必须同时清理 sender、token/candidate metadata 和 manager room 状态。

## 9. Decode executor

`DisaggDecodeExecutor` 在请求注册时创建 `MooncakeKVReceiver`。scheduler admission
产生 `Forward.Batch` 后：

1. 要求 remote-prefill batch 不与普通 local batch 混合；
2. 从每行 block table 构造并验证 destination manifest；
3. receiver 发布 pre-allocation；
4. transfer success 后取得 bootstrap token 与可选 speculative candidates；
5. 生成 `PD.RemotePrefillDoneEvent`；
6. event loop 将 remote cache slot/runtime state 交给首次本地 decode。

Decode 从未本地执行 prompt，因此 `reset_valid_cache_length()` 必须用完整 prompt
length 初始化 runtime row；state snapshot group 也依赖该长度选择正确 block。

## 10. C++ FSM 回灌

Python executor 生成的事件进入 `Scheduler.advance()`：

- `BootstrappedEvent`：bootstrap 完成，可继续 P/D 协调；
- `RemotePrefillDoneEvent(request_id, bootstrap_token)`：Decode 侧 prompt cache 已可用，
  把首 token 接入 token container 并进入 decode；
- `SucceededEvent`：Prefill 侧发送完成；
- `FailedEvent`：终止/回收请求。

cache transfer 期间 scheduler pin 对应请求，避免 destination/source block 被 prefix
eviction 或 retraction 提前回收。

## 11. 维护检查表

- 传输 layout 必须来自 `CacheArena` owner，不能从 pool 拼 pointer list；
- 新 field dtype/shape 先进入 recipe/plan，再更新 TP partition schema；
- 新 group 必须定义 PD transfer policy；
- manifest 只传 block id 与逻辑窗口，不复制字节 layout；
- peer validation 必须在发布任何远端等待状态之前完成；
- terminal success/failure 必须清理所有 request/room/session state；
- 改动 wire dataclass 时同步大小限制、parser、round-trip 与跨 TP 测试。
