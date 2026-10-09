# tokenspeed-kernel 设计架构

`tokenspeed-kernel/` 是 TokenSpeed 的算子接口、实现注册与选择层。它把模型需要的
计算语义与具体硬件实现分开：runtime 调用算子接口，kernel 层根据数据格式、设备能力
和算子约束选择实现，再启动设备计算。该包也可以独立使用，不依赖 TokenSpeed 的请求
调度器。

本文按当前源码解释各层职责、一次调用的路径以及扩展方法。具体算子的布局、数值公式
和调优细节放在对应 `ops/` 目录；整体请求路径见 [Inference Flow](inference-flow.md)。

## 总览

```text
TokenSpeed runtime：模型层 / attention backend / sampling
                         │ 算子输入、cache view、执行参数
                         ▼
tokenspeed_kernel.ops.<family>[.<variant>]：公共接口
  ├─ 整理输入与格式，表达所需 traits / features
  ├─ select_kernel(...) ────────────────┐
  └─ ShapeCapture / kernel_scope       │ 查询
                                       ▼
                              KernelRegistry
                         KernelSpec + Python callable
                                       ▲
                                       │ @register_kernel(...)
              ┌────────────────────────┼──────────────────────────┐
              │                        │                          │
       通用 Triton 实现        硬件专用实现 / 适配层          外部插件
              │                        │                    显式发现并注册
              │              ┌─────────┼──────────┐
              │              │         │          │
              │           CuTe DSL    AMD        Ascend
              │           第三方库    Gluon      NPU 算子
              └──────────────┴─────────┴──────────┘
                                       │
                                       ▼
                                  设备 kernel

numerics / benchmark：复用注册信息，进行数值验证与性能测量
compile_monitor：观察 Triton / Gluon JIT 编译与 serving 阶段的重新编译
```

这张图描述公共分派路径。部分算子还包含预先准备的执行计划或专用快速分派，例如
BF16 GEMM 的联合调优路径；不能把“经过 kernel 包边界”理解为“每次都重新遍历注册表”。

## 1. 包边界与目录

### 1.1 三层职责

| 层 | 负责什么 | 边界 |
| --- | --- | --- |
| TokenSpeed runtime | 模型结构、batch metadata、cache 生命周期、graph capture 与执行编排 | 通过 `tokenspeed-kernel` 使用算子，不直接依赖第三方 kernel 包 |
| `tokenspeed-kernel` | 公共算子语义、格式描述、注册、选择、实现适配、验证和测量 | 接收 runtime 准备的 tensor/view，不接管请求 admission、prefix matching 或 cache block 分配 |
| 具体实现 | 按算子契约读写 tensor，执行设备计算 | 硬件、布局和 shape 限制要在接口校验及注册约束中表达 |

这里的 runtime attention backend 与 kernel solution 不是同一层。前者把执行计划和
cache view 转成一次 attention 调用；后者实现这次调用中的计算。新增 per-request
attention state 时，应先评估能否由 LCM cache group 与 C++ scheduler 管理，而不是
在某个 solution 内另建请求状态表。

### 1.2 源码布局

```text
tokenspeed-kernel/
├── README.md                         # 高层设计与使用入口
├── python/
│   ├── pyproject.toml                # Python 包构建入口
│   └── tokenspeed_kernel/
│       ├── __init__.py               # profiling 环境初始化
│       ├── platform.py               # 设备描述与能力要求
│       ├── signature.py              # tensor/scale 格式契约
│       ├── registry.py               # 注册信息、实现与选择结果缓存
│       ├── selection.py              # 过滤、排序、override 与诊断
│       ├── _triton.py                # Triton / Gluon 导入适配
│       ├── profiling.py              # shape capture 与 profiling scope
│       ├── compile_monitor.py        # JIT 编译监控与 serving 标记
│       ├── ops/
│       │   ├── attention/            # mha / mla / dsa / kda / prologue 等
│       │   ├── gemm/                 # mm / bmm 与投影算子
│       │   ├── moe/                  # routing / dispatch / experts / combine
│       │   ├── residual/             # norm、残差与融合操作
│       │   └── ...                   # embedding / quantization / sampling 等
│       ├── thirdparty/               # 第三方实现与适配代码
│       ├── plugins/                  # 外部包 entry point 发现
│       ├── numerics/                 # 输入、reference、容差与比较
│       └── benchmark/                # generator、计时、报告与 CLI
├── benchmarks/                       # 按 vendor / arch 组织的测量 suite
└── test/                             # 公共基础设施与算子测试
    ├── amd/                          # AMD 专用测试
    └── nvidia/                       # NVIDIA 专用测试
```

公共 API 主要位于各 `ops` 子包，实际导入方式例如：

```python
from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
from tokenspeed_kernel.ops.gemm import mm
```

当前顶层 `tokenspeed_kernel/__init__.py` 只做 profiling bootstrap，并未统一重导出这些
算子。导入算子子包时，子包末尾的实现模块导入触发注册 decorator；工具需要批量加载
内置实现时使用 `registry.load_builtin_kernels()`。仅导入顶层包不会加载所有实现。

## 2. 用哪些对象表达一个算子

注册定义位于 [registry.py](../tokenspeed-kernel/python/tokenspeed_kernel/registry.py)，
数据格式定义位于 [signature.py](../tokenspeed-kernel/python/tokenspeed_kernel/signature.py)。

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| `family` / `mode` | 算子族及其具体操作，是注册表的主索引 | `attention.mha_decode_with_kvcache`、`gemm.mm` |
| `solution` | 实现方案，同一方案可有多个注册项 | `triton`、`gluon`、`reference` |
| `name` | 某个具体注册项的唯一名字 | `gluon_mha_decode_gfx950` |
| `FormatSignature` | 按逻辑 tensor role 描述的一组精确格式 | `q` / `k_cache` / `v_cache`，或 `a` / `b` |
| `CapabilityRequirement` | 对厂商、架构版本、硬件及运行时能力的要求 | AMD 架构范围、所需 tensor-core 能力 |
| `features` | 实现声明的功能集合，调用方可以要求其子集 | `batch_invariant` |
| `traits` | 非格式类的适用条件 | `head_dim`、`page_size`、`return_lse`、shape 对齐 |
| `priority` | 候选项排序所用的优先级 | `Priority.PORTABLE`、`Priority.SPECIALIZED` |

### 2.1 为什么不能只用 dtype 选择

同一个存储 dtype 可以有不同的数学含义。例如，FP8 tensor 可以是无 scale 的原始
值，也可以配合 per-tensor 或 per-block scale；打包在 `uint8` 中的权重也可能采用
不同的低比特格式。因此格式分成三层：

```text
FormatSignature
  └─ role → TensorFormat
               ├─ storage_dtype：主 tensor 的物理 dtype
               ├─ format：dense / scaled-fp8 / mxfp8 / mxfp4 / nvfp4 等
               └─ scale → ScaleFormat
                              ├─ scale 的存储 dtype
                              ├─ granularity
                              └─ block_shape 或 dynamic_block_shape
```

一个 signature 为每个 role 指定一种格式；支持多个组合时注册多个 signature。
选择器按完整 signature 匹配，不把所有使用同一 dtype 的实现视为兼容。stride、head
dimension、page size 等条件则由 traits 和算子入口校验表达，不属于格式名称本身。

### 2.2 注册表保存什么

`@register_kernel(...)` 把 `KernelSpec` 与 Python callable 写入进程内的
`KernelRegistry` 单例。注册表同时维护 `(family, mode)` 索引、名称索引、实现映射
以及选择结果缓存。重新注册同名 kernel 会替换旧项，并使相关算子的选择缓存失效。

`KernelSpec.weight_preprocessor` 可以关联权重预处理函数。需要重排或打包权重的
算子由自己的 planning/preparation 接口调用它，例如 MoE 的 plan 与权重准备流程。
注册动作以及通用 `select_kernel()` 本身不会自动转换权重。

## 3. 从候选项到可调用实现

[selection.py](../tokenspeed-kernel/python/tokenspeed_kernel/selection.py) 中的
`select_kernel()` 返回 `SelectedKernel`，它既能像函数一样调用，也暴露最终注册名
`name`，供日志、profiling 和测量记录使用。

### 3.1 普通选择路径

1. 获取 `PlatformInfo`，解析是否有有效 override。
2. 没有 override 时，按算子、格式、架构、features、traits 和 solution 查询缓存。
3. 从注册表取同一 `(family, mode)` 的候选项，过滤设备能力、格式、features，以及
   调用方指定的 solution。
4. 过滤算子 traits 和 shape 条件，例如 head dimension、GEMM 对齐、上下界或
   `mnk_problem_filter`。
5. 对候选项按 `(oracle score, priority)` 从高到低排序。
6. 返回获胜实现，并缓存 `SelectedKernel`；无候选项时抛出 `NoKernelFoundError`。

`SelectionOracle` 提供每个 family 的领域评分；没有 oracle 时使用中性分。
**oracle score 优先于 priority**，因此不能仅根据优先级数字推断最终实现。

普通 trait 匹配对注册项未声明的 trait 通常视为不限制；shape 边界和部分尺寸条件另有
更严格的规则。因此，一个 kernel 不支持的语义必须明确声明限制，不能依靠“没写这个
trait”排除自己。`features` 则要求调用方请求的集合是实现声明集合的子集。

### 3.2 Priority 的设计分段

| 分段 | 数值 | 用途 |
| --- | --- | --- |
| `REFERENCE` | 0 | 数值参考实现 |
| `PORTABLE` | 4–7 | 通用覆盖实现 |
| `PERFORMANT` | 8–11 | 较广适用范围的优化实现 |
| `SPECIALIZED` | 12–15 | 有明确架构或 shape 限制的专用实现 |
| `PLUGIN` | 16–19 | 为外部插件预留的优先级空间 |

priority 用于已通过兼容性过滤的候选项；提高 priority 不会让不兼容的实现获得支持。
reference 的设计用途是数值基准，但选择器没有通用的“排除 reference”步骤，其是否
成为候选仍取决于已加载注册项和相同的过滤、排序规则。

### 3.3 `solution` 与 `override`

| 入口 | 行为 |
| --- | --- |
| `solution="gluon"` | 限定实现方案，保留正常的能力、格式、traits 检查和排序 |
| `override="具体注册名"` | 强制取得指定项，跳过能力、格式和 traits 匹配；仍检查所需 features |
| `override="方案名"` | 按方案解析注册项，优先取声明所需 features 的最高优先级项，同样绕过普通兼容性过滤 |

override 的实际优先顺序是：

```text
TOKENSPEED_KERNEL_OVERRIDE_<FAMILY>_<MODE>
  → kernel_override(family, mode, kernel_name) 上下文
  → 本次调用的 override 参数
```

override 用于定向实验和排障，不能作为输入兼容性证明。普通选择失败也不会自动转换
输入格式，或在 kernel 执行失败后捕获异常并遍历其他候选项。

### 3.4 选择缓存、调优与 JIT 是三件事

选择缓存保存的是 Python callable，命中后省去过滤和排序；它不等于设备代码已经
编译或 CUDA graph 已经 capture。

当前通用 `SelectionStrategy.AUTOTUNE` 虽然已有接口，但 `_autotune_select()` 实际
仍回退到启发式排序，没有在这里测量候选项。GEMM 等算子自身可以有独立的调优与快速
分派流程；例如 `mm()` 中的 joint BF16 路径，仍由公共 kernel 包管理，并保留 shape
记录与 profiling。阅读性能路径时应同时检查算子入口与通用选择器。

## 4. 一次 MHA decode 调用

以 [ops/attention/mha/__init__.py](../tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/__init__.py)
中的 `mha_decode_with_kvcache()` 为例：

```text
runtime 准备 q、KV cache view、page_table、cache_seqlens
  → 构造 q / k_cache / v_cache 的 FormatSignature
  → 提取 q_len、head_dim、page_size、sliding_window、sinks 等 traits
  → select_kernel("attention", "mha_decode_with_kvcache", ...)
  → ShapeCapture 记录输入 shape 和选中的注册名
  → kernel_scope 包围一次算子执行
  → SelectedKernel(...) 调用具体 solution 的 launcher
  → launcher 启动设备 kernel 并返回结果
```

这里的 `max_seqlen_q` 普通 decode 为 1，也可以表达 compact speculative decode 的
多 query token 情形；它是同一接口的输入。公共接口依据 scale 参数区分 dense 与
MXFP8 格式，并将所需 scale 继续传给实现。

这次调用消费已经准备好的 cache。对于 `PagedAttention` 层，新 K/V 的 norm、RoPE、
量化和写入由 attention prologue 统一处理，core attention backend 不重复写入。
GQA 和 MLA 各有 prologue 入口；DeepSeek V4/V4.1 等独立 attention 路径的边界见
[attention-prologue 设计](../docs/design/attention-prologue.md)。

cache 的 prefix identity、block 分配与 kernel page view 分属不同层，见
[cache-concepts](../docs/design/cache-concepts.md)。eager 与 CUDA graph decode 共用
metadata 的刷新契约，graph 的差异留在执行编排层，见
[unified_path](../docs/design/unified_path.md)。

## 5. 多硬件实现与插件

### 5.1 实现文件的组织

实现按 `ops/<family>/<solution>` 组织；attention 再增加 variant，例如
`ops/attention/mha/triton.py`。多文件实现的 helper 放在对应私有目录。
公共 API、注册 adapter 和底层设备代码可以分文件，但仍属于同一个算子契约。

- 通用实现优先使用 Triton；NVIDIA 优化实现优先采用 CuTe DSL，AMD 优化实现优先采用
  Triton Gluon。第三方库通过 `thirdparty/` 适配后，由 `ops/` 注册。
- [`_triton.py`](../tokenspeed-kernel/python/tokenspeed_kernel/_triton.py) 集中处理
  `tokenspeed_triton`、Gluon、Proton 等导入；Ascend 的 Triton 接口通过 NPU 包适配。
  kernel 包内其他模块应通过这个入口导入，避免混用两套 Triton 类型和 JIT 状态。
- [`tokenspeed-kernel-amd`](../tokenspeed-kernel-amd/README.md) 按架构和算子组织
  Gluon 实现，不反向依赖 `tokenspeed-kernel`。例如 MHA 的 `gluon.py` adapter 在
  kernel 主包中声明 `KernelSpec`，再调用 AMD 包的 `launch_*` 函数。
- [`tokenspeed-kernel-npu`](../tokenspeed-kernel-npu/README.md) 保存 Ascend 算子
  与 Triton-Ascend 适配，主包负责注册和分派。runtime 仍从主包调用算子；存在 NPU
  adapter 不意味着所有算子或模型都获得了 NPU 支持。

设备能力由 [platform.py](../tokenspeed-kernel/python/tokenspeed_kernel/platform.py)
中的 `PlatformInfo` 与 `CapabilityRequirement` 匹配。注册声明与模块是否成功加载
共同决定候选集；相同函数名、同样的 GPU 厂商或已安装一个依赖，都不足以证明某个
输入能够使用该实现。

### 5.2 外部插件

插件包通过 `tokenspeed_kernel.plugins` entry-point group 暴露 `register()`，在其中
使用同一个 `register_kernel` 注册实现。独立宿主需要显式加载内置实现并发现插件：

```python
from tokenspeed_kernel.plugins import discover_plugins
from tokenspeed_kernel.registry import load_builtin_kernels

load_builtin_kernels()
discover_plugins()
```

仅安装插件或导入顶层包不会触发发现。TokenSpeed runtime 的插件初始化会调用 kernel
插件发现；独立脚本需自行安排初始化。插件发现按 entry-point 名排序，加载失败会
警告并继续处理其他插件；`TOKENSPEED_KERNEL_DISABLE_PLUGINS` 可以禁用指定名字。

插件同样受格式、能力和 traits 约束，priority 也仍排在 oracle score 之后。插件契约
当前是实验性的，包版本和完整使用方式见
[插件说明](../tokenspeed-kernel/python/tokenspeed_kernel/plugins/README.md)。

## 6. 验证、性能与可观测性

### 6.1 数值验证

[`numerics/`](../tokenspeed-kernel/python/tokenspeed_kernel/numerics/) 将输入生成、
reference 实现、容差策略和输出比较分开。验证器按 signature 和 traits 寻找兼容的
reference；缺少对应 reference 或输入生成器，不能被解释成该 kernel 已验证正确。
容差应依据算子数值契约设置；要求逐位一致的路径不能用宽松 `atol` / `rtol` 代替。

在已安装 kernel 包及目标设备依赖的环境中，可从仓库根目录按算子运行：

```bash
python -m tokenspeed_kernel.numerics \
  --op attention.mha_decode_with_kvcache --dtype bf16 --dtype-role q
```

`--dtype-role` 是必填项，因为同一 signature 中 query、权重和 scale 可能具有不同
存储 dtype。模型级数值承诺另见 [numerics 设计](../docs/design/numerics.md)；单算子
reference 比较不能替代完整模型的数值验证。

### 6.2 性能测量

`benchmark/` 提供 CLI 以及 registration-level `KernelBenchmarkHarness`。
后者将职责拆开：算子 generator 准备输入、调用与可选正确性校验，共用 timer 负责
warmup、graph replay 和设备计时，报告记录实际注册名、原始样本及失败状态。

```bash
python -m tokenspeed_kernel.benchmark \
  --op attention.mha_decode_with_kvcache --dtype bf16 --dtype-role q --verify
```

对于 graph-replay harness，输入生成、选择、JIT 编译、capture 和验证不计入报告的
设备执行时间；正确性校验由 generator 显式启用，并非每个 case 都默认验证。
跨版本比较需要保持硬件、case 定义、warmup 和 cache mode 可比。suite 格式和
测量口径以 [benchmark 说明](../tokenspeed-kernel/benchmarks/README.md) 为准。

### 6.3 Shape 与 profiling

[`profiling.py`](../tokenspeed-kernel/python/tokenspeed_kernel/profiling.py) 提供：

- `ShapeCapture`：记录算子、注册名、dtype 和 shape，供工作负载分析及重放工具使用；
- `kernel_scope`：为算子执行附加 scope 与 metadata；
- Proton 初始化和环境配置：控制 profiling session 及输出。

选择排障先使用 `explain_selection()` 查看候选、排除原因、评分及有效 override，或
开启 `TOKENSPEED_KERNEL_VERBOSE=1` 观察分派结果。需要核对单个注册项时使用
`describe_kernel(name)`。确认选中了什么之后，再分析设备 trace 与性能数据。

## 7. JIT 与 serving 的边界

选择结果缓存不能消除编译缓存的新 key。`tl.constexpr`、`gl.constexpr` 和
`cutlass.Constexpr` 会把参数值纳入 JIT 专门化；每批变化的 token 数、request 数、
sequence length、block-table width 及派生 stride 不应直接成为无限增长的编译维度。

模型维度、固定功能开关和少量必要性能参数可以在启动时专门化。动态维度优先作为
runtime 参数；确需编译期边界时，要限制分桶并覆盖启动预热，避免 serving 遇到新桶
再编译。能用固定 block 加循环表达的动态范围，不应每换一个 batch 就生成新代码。

普通 runtime 参数也可能因整数是否等于 1、是否可被 16 整除，以及指针对齐等 Triton
专门化规则产生新 key。按 batch 变化的计数应评估 `do_not_specialize`，按动态偏移切片的指针应评估
`do_not_specialize_on_alignment`。保留哪些对齐信息，应由实际访问方式决定。

[`compile_monitor.py`](../tokenspeed-kernel/python/tokenspeed_kernel/compile_monitor.py)
观察 Triton JIT，Gluon 共用该机制。runtime 在 worker 初始化时安装监控，在进入
serving 时调用 `mark_serving()`；随后记录编译耗时、key 变化及调用位置，对持续产生
新值的编译期参数按 `TOKENSPEED_JIT_COMPILE_CHECK=warn/error` 警告或报错。
该监控不覆盖 DeepGEMM、CuTe DSL 等其他 JIT，不能据此声称所有实现都没有重新编译。

新增动态 shape 路径时，应使用 `test/utils.py` 的 `assert_no_triton_compile` 验证
预热后改变相关 scalar 不会再编译，并覆盖 eager/graph 实际调用。启动时的编译、
选中实现的正确性与 serving 阶段的稳定性是不同的验证项。

## 8. 扩展与读代码顺序

增加一种实现时，按以下顺序推进：

1. 先明确现有公共接口的输入、输出、布局、可选功能和数值契约，复用同一执行路径。
2. 在对应 family/variant 下增加 solution；第三方代码归入 `thirdparty/`，硬件专用
   实现使用可识别的 solution 文件名或目录。
3. 注册精确 signatures、capability、features、traits 和合理 priority；注册名与
   Python 函数名保持一致并包含 solution 前缀。确认子包会导入新模块并触发注册。
4. adapter 必须保留调用方显式传入的参数；不支持的语义要显式拒绝，不用 `**kwargs`
   静默吞掉。新增决定执行路径或正确性的参数必须显式给出。
5. 增加契约、数值和必要的动态 scalar 编译测试；公共测试放 `test/`，厂商专用测试放
   对应 vendor 子目录。需要性能证据时补充 generator/suite 并在目标设备测量。
6. 在算子目录更新详细文档；跨层不变量变化时同步修改相应 `docs/design/` 文档。
   创建 commit 前运行仓库要求的 `pre-commit run --all-files`。

推荐阅读路线：

| 问题 | 源码入口 |
| --- | --- |
| 一次算子调用怎样被分派？ | `ops/attention/mha/__init__.py` → `selection.py` → `registry.py` |
| 为什么某个 dtype/shape 不匹配？ | `signature.py` → `platform.py` → 目标 solution 的注册声明 |
| AMD/NPU 实现怎样接入？ | `ops/attention/mha/gluon.py`、`ops/attention/mha/triton.py` → 对应厂商包 |
| 权重布局何时转换？ | `ops/moe/__init__.py` 的 plan/prepare 流程 → `KernelSpec.weight_preprocessor` |
| 怎样验证和复现性能？ | `numerics/verify.py` → `benchmark/` → `benchmarks/README.md` |
| 为什么 serving 中突然编译？ | `compile_monitor.py` → launcher 参数与 JIT 签名 |

注册与选择规则的测试入口包括
[`test_registry.py`](../tokenspeed-kernel/test/test_registry.py)、
[`test_selection.py`](../tokenspeed-kernel/test/test_selection.py)、
[`test_kernel_api_selection.py`](../tokenspeed-kernel/test/test_kernel_api_selection.py)
及 [`test_plugins.py`](../tokenspeed-kernel/test/test_plugins.py)。它们与设备算子的正确性、
benchmark 和端到端 serving 测试分别覆盖不同层次的契约。
