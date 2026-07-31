
## 8. FlatKV PD 契约

flat-state / LCM two-level 布局（见 `kvcache-management.md` 的 LCM 章节）下，KV 不再是
「逐层连续 buffer + 页索引向量」，而是多个 cache group 共享一个页 ID 命名空间、绑到别
名化的原始 slab。prefill / decode 各自独立分配页，所以两边要交换**结构化、带版本、可
交叉校验**的 layout + manifest。这套契约（model-neutral）在 `flatkv.py`，协议版本
`FLATKV_PD_PROTOCOL_VERSION = 1`（`:37`）。

关键类型：
- **`FlatKVPDLayout`**（`:345`）：一侧本地的完整 flat cache ABI + 容量——`version`、
  `layout_fingerprint`（64 位 SHA-256 hex）、`block_size`、`num_pages_with_null`、
  `physical_buffer_ids`、`physical_page_bytes`、`groups`。`__post_init__`（`:357`）做大
  量不变量校验（slot 覆盖、segment 不越界、无重复 group）。
- **`FlatKVPDPeerLayout`**（`:302`）：peer 只需的子集（`version` +
  `layout_fingerprint` + `num_pages_with_null`），跨节点比对 ABI 用；
  `to_wire_bytes`/`from_wire_bytes`（`:318` / `:325`）走 canonical JSON。
- **`FlatKVPDGroup`**（`:223`）：一个 cache group——`group_id`、`family`
  （`history`/`state`）、`transfer_policy`（`full_suffix`/`latest_snapshot`）、
  `physical_slots`、`cache_blocks_per_lcm_block`、`transfer_segments`（每个是
  `FlatKVPDTransferSegment` `:199`，描述一个 strided 字段拷贝）。
- **`FlatKVPDPageManifest`**（`:457`）：一个请求的逐 group 选页——`groups`（每个
  `FlatKVPDGroupPages` `:438`：`group_id` + `page_ids`）+ `prefix_len` + `prompt_len` +
  `version`。同样 `to_wire_bytes`/`from_wire_bytes`（`:487` / `:494`）。

关键函数：
- **`build_flatkv_page_manifest(forward_op, layout, request_row, prefix_len,
  prompt_len)`**（`:671`）：从 scheduler 的 `flat_block_tables_arrays()` 按各 group 的
  `transfer_policy` 选页（`full_suffix` 选 `[prefix, prompt)` 全后缀，
  `latest_snapshot` 只选最后一块，`_logical_slots` `:545`），并校验没有别名 live 未选
  slot（`:773`-787）。prefill 的 `_flat_decode`、decode 的 `_flat_prefill` 都用它。
- **`validate_flatkv_manifest_pair(src, dst, layout, dst_num_pages_with_null)`**
  （`:621`）：收发双方 manifest 交叉校验——group 顺序、页对齐、页数符合 policy、
  prefix_len/prompt_len 一致（`:640`-646）。
- **`validate_flatkv_slab_registrations(...)`**（`:791`）：校验原始 slab 注册与 layout
  一致（数量、slot 顺序、buffer_id、extent、无重叠）。
- **`build_lcm_flatkv_pd_contract(plan, backing, group_specs, field_dtypes)`**
  （`:844`）：从一个 LCM arena 造 `(FlatKVPDLayout, registrations)`——不拷贝不摊平，直接
  按 arena 的整数几何算 `transfer_segments` 的 `page_zero_offset`/`page_stride_bytes`
  （`:886`-903），并对整个 plan 算 SHA-256 `layout_fingerprint`（`:915`-963）。

契约的产出方是 KV pool：`LcmCachePool.pd_contract(group_specs)`
（`layers/attention/kv_cache/lcm.py:80`）内部调 `build_lcm_flatkv_pd_contract`
（`:94`）；`LcmMHATokenToKVPool.get_flatkv_pd_contract`
（`kv_cache/lcm_mha.py:212`）/ `LcmMLATokenToKVPool.get_flatkv_pd_contract`
（`kv_cache/lcm_mla.py:183`）是 `get_kv_args` 走的 ABI 入口。

## 9. 异构 TP 的传输分片规划器（transfer_plan.py）

当 prefill 和 decode 的 TP size 不同（或 GQA/MQA 下 KV head 数少于 TP 数），一个 decode
rank 需要的某个 KV 分片可能散落在多个 prefill rank 上，反之亦然。`PDTransferPlanner`
（`transfer_plan.py:204`）为一个 decode rank 算出该从哪些 prefill rank、拷哪些字节区
间。

- 输入：`ParallelLayout`（`:41`，role/world_size/dp_size）、每个 buffer 的
  `BufferLayout`（`:62`，`logical_axis`（kv_head/state_channel/replicated）/
  `logical_size`/`item_stride_bytes`/`tp_replica_group_size` 等）。
- **`plan_for_decode_rank(decode_rank)`**（`:221`）：TP 相等且无 replica 分组时走
  **identity plan**（`:229`-246，一对一，`plan_kind="identity"`），否则对每个 buffer
  按逻辑区间求交（`_rank_interval_for_buffer` `:450`，`intersect` `:196`）算出
  `TransferFragment`（`:93`，src/dst rank + 字节偏移 + 每页字节数），产出
  `plan_kind="fragmented"` 的 `RankTransferPlan`（`:174`）。
- `_calc_source_fanout`（`:387`）预算每个 prefill rank 要响应多少 decode rank
  （`required_dst_info_num`，即 §7.2 里凑齐才 `Bootstrapped` 的那个数）。
- `TransferFragment` 经 `encode_transfer_fragments`/`decode_transfer_fragments`
  （`:110` / `:134`，`TRANSFER_PLAN_PROTOCOL_VERSION=1` `:107`）序列化进预分配 ZMQ 消息
  的第 12/13 帧（`entities.py:168`-170）。

`MooncakeKVReceiver._calc`（`receiver.py:378`）是调用点：FlatKV（`:384`）、MLA-L1.5
（`:417`）、非-MLA 经 `PDTransferPlanner`（`:429`-430）、MLA 等/大/小 TP 分支
（`:432`-466），产出 `ReceiverRoutePlan`（`:131`）。

## 10. MOONCAKE_ASYNC 异步后端现状

`MooncakeAsyncKVManager`（`async_conn.py:70`）是走 Mooncake 异步 submit/poll API
（`transfer_submit_write` → `batch_id` → `transfer_check_status`）而非同步 batch write
的另一套 prefill 管理器，核心是层流水状态机 `async_transfer_worker`（`:265`，内含
`discard_finished_bid_inplace` `:266` 解码状态码 `1`=done/`-1`=timeout/`-2`=failed、
`submit_transfer` `:374`、`pop_transferred` `:458`、`finalize` `:504`）。

> ⚠️ **当前该路径与重构后的同步路径不同步，构造即失败。**
> `MooncakeAsyncKVManager.__init__`（`:71`-81）签名是
> `(args, disaggregation_mode, server_args, is_mla_backend, draft_is_mla_backend)`，并
> `super().__init__(...)` 用同样 5 参调父类；但它的父类经 `import ... as
> MooncakeKVManager`（`:37`）其实是 `MooncakeKVManagerPrefill`，后者当前签名只有
> `(args: KVManagerArgs, kv_args: KVArgs)`（`prefill.py:73`）。它还引用
> `args.offsets`（`:92`）、`server_args.kv_cache_quant_method`（`:87`），都对不上现在
> 的 `KVArgs`/`KVManagerArgs`。这说明 `MOONCAKE_ASYNC` 是**遗留/未维护**分支，不要当成
> 和同步 `MOONCAKE` 对等的可用后端。新代码走 `MOONCAKE`。

## 关键数据结构

| 数据结构 | 位置 | 作用 |
| --- | --- | --- |
| `DisaggPrefillExecutor` / `DisaggDecodeExecutor` | `prefill_executor.py:48` / `decode_executor.py:44` | PD 两侧每步编排 + C++ 事件泵；持每请求 sender/receiver。 |
| `MooncakeKVManagerPrefill` / `MooncakeKVManagerDecode` | `mooncake/prefill.py:72` / `decode.py:78` | 每引擎一个的长期管理器：ZMQ 收线程 + 传输 worker / 心跳。 |
| `MooncakeKVSender` / `MooncakeKVReceiver` | `mooncake/sender.py:35` / `receiver.py:469` | 每请求一个的发送/接收句柄。 |
| `MooncakeTransferEngine` | `base/mooncake_engine.py:26` | 包 `mooncake.engine.TransferEngine`，单边 RDMA 写。 |
| `DisaggBootstrapServerBase` / `MooncakeKVBootstrapServer` | `base/bootstrap.py:42` / `mooncake/conn.py:93` | HTTP rendezvous（prefill PUT ip:port，decode GET）。 |
| `DisaggManagerBase` | `base/manager.py:31` | engine + ZMQ 控制 socket + room 键控状态 FSM。 |
| `TransferPoll` | `base/status.py:22` | 6 个状态常量（Failed 粘滞、单调递增）。 |
| `KVArgs` / `KVManagerArgs` | `mooncake/entities.py:39` / `:242` | buffer 描述符 / 引擎级配置。 |
| `TransferInfo` / `KVArgsRegisterInfo` / `TransferKVChunk` | `mooncake/entities.py:110` / `:204` / `:85` | 预分配 / 指针注册 / 发送工作项的 wire 结构。 |
| `PD.{Bootstrapped,Failed,Succeeded,RemotePrefillDone}Event` | `tokenspeed-scheduler/csrc/scheduler/outside_events/pd.h:30`-`:54` | C++ PD 事件 variant。 |
| `FlatKVPDLayout` / `FlatKVPDPageManifest` | `flatkv.py:345` / `:457` | FlatKV 的结构化 layout / 逐请求选页 manifest。 |
| `PDTransferPlanner` / `TransferFragment` | `transfer_plan.py:204` / `:93` | 异构 TP 的传输分片规划。 |
| `BootstrapInfo` | `base/bootstrap.py:35` | 逐请求 rendezvous 三元组（host/port/room）。 |
| `MetadataBuffers` | `utils.py:288` | 首 token metadata 的 RDMA buffer（output_ids/logprobs）。 |
| `StepCounter` | `utils.py:433` | layerwise 传输的 cache/aux step 计数（CUDA event fence）。 |

## 读代码建议

想快速掌握 PD 主线，按此顺序读：

1. `python/tokenspeed/runtime/utils/server_args.py:635`（`resolve_disaggregation`：三种
   模式各自约束）
2. `python/tokenspeed/runtime/pd/factory.py:188`（`create_kv_transfer` +
   `get_kv_args`：造 Executor 与 buffer 描述符）
3. `python/tokenspeed/runtime/engine/event_loop.py:886`（`_dispatch_forward` 四条
   path——PD 的编排主干）
4. `python/tokenspeed/runtime/pd/decode_executor.py:159` 和
   `prefill_executor.py:244`（一次预分配 / 一次发 KV 的具体索引计算）
5. `python/tokenspeed/runtime/pd/mooncake/prefill.py:915`（`transfer_worker`：真正搬 KV
   的循环）+ `mooncake/receiver.py:601`（decode 注册指针）
6. `tokenspeed-scheduler/csrc/fsm/pd_events.cpp:40`（首 token 如何注入 decode 请求的
   FSM）

想深入特定子系统：

- **控制面 rendezvous / ZMQ 协议**：`base/bootstrap.py`、`mooncake/conn.py`、
  `mooncake/entities.py` 的 `from_zmq`。
- **数据面 RDMA**：`base/mooncake_engine.py`、`mooncake/prefill.py`（`send_kvcache` /
  `send_kvcache_layerwise`）、`mooncake/decode.py`（`_handle_prefill_status`）。
- **FlatKV / LCM PD**：`flatkv.py`、`layers/attention/kv_cache/lcm.py:80`
  （`pd_contract`）、`lcm_mha.py` / `lcm_mla.py` 的 `get_flatkv_pd_contract`。
- **异构 TP**：`transfer_plan.py`、`mooncake/receiver.py:378`（`_calc`）。
- **C++ 事件与 FSM**：`tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`、
  `fsm/pd_events.cpp` / `pd_events.h`、`scheduler/outside_events/pd.h`。
