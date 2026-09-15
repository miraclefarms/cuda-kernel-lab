# 01-execution-model

执行模型与三机基线画像。`bench-probe` 是三台机器共用的基线画像工具，也是所有带宽图达成率
分母的来源。

## 要验证的问题

- 三台机器（a100 / h20 / h200）的算力:带宽比（ridge point）到底差多少。
- 预期：h20 与 h200 同 ISA、几乎同带宽，只有算力差 6.7×，ridge point 差 6.6×——同一个优化
  可能在两台上收益反号。
- 本篇不做优化对比，只把「争论任何优化之前必须先钉死的条件」量出来。

## 目录代码

| 文件 | 作用 |
|---|---|
| `probe_main.cu` | `bench-probe`：打印 device props 与特性门，并实测 streaming 上限（read / copy / write，cold / hot）|
| `CMakeLists.txt` | 产出目标 `bench-probe` |

**为什么没有 baseline / optimized**：本篇是基线画像，不是「同一问题两份实现比收益」，因此
只有一个可执行文件；其余篇目才有并列实现。

## 所用技术

- `cudaDeviceGetAttribute`：CUDA 13 已从 `cudaDeviceProp` 移除 clockRate / memoryClockRate /
  computeMode，只能通过 attribute 读取
- 特性门：cc ≥ 9.0 → TMA / cluster / DSMEM / wgmma；cc ≥ 10.0 → tcgen05 / TMEM
- 与 kernel 同一套口径的 streaming 测量：`bench::measure_stream_ceiling`（cold / hot，工作集
  `bench::kStreamBufferBytes`），所以这个上限可以直接当带宽类篇目的分母

## 实验数据

数据来源：

- h20 — `results/01-execution-model/2026-09-14-h20.csv`（commit `dfa8153`）
- h200 — `results/01-execution-model/2026-09-15-h200.csv`（commit `b56c035`）
- a100 — **无有效数据**，见下文「a100 占位」

口径：256 MiB 缓冲、fp32、cold/hot 分开、30 次采样报分位数（下表取 median）。两卡标称带宽
同为 **4814 GB/s**（`bench/machine-peaks.json`）。

> **这次能说什么，先说清楚**：h20 一轮采于非独占机器（8×vLLM worker 常驻）且未锁频，按
> `docs/measurement-methodology.md` 只能作同会话相对比值；h200 一轮的独占性/锁频状态尚未
> 回填 `docs/environment-matrix.md`，同样只作相对比值。因此下表的「% 标称」仅用于两卡横向
> 比较，**不作为可引用的绝对上限**；要报绝对值，先补齐环境矩阵并锁频重采。

### 实测带宽（median）

| variant | 口径 | h20 GB/s | h20 %标称 | h200 GB/s | h200 %标称 | h200/h20 | a100 GB/s | a100 %标称 |
|---|---|---|---|---|---|---|---|---|
| read | cold | 3317.6 | 68.9% | 3559.0 | 73.9% | 1.073 | 待采集 | 待采集 |
| read | hot | 3954.2 | 82.1% | 4263.4 | 88.6% | 1.078 | 待采集 | 待采集 |
| copy | cold | 3775.7 | 78.4% | 3899.0 | 81.0% | 1.033 | 待采集 | 待采集 |
| copy | hot | 3928.8 | 81.6% | 4042.3 | 84.0% | 1.029 | 待采集 | 待采集 |
| write | cold | 4046.6 | 84.1% | 4111.1 | 85.4% | 1.016 | 待采集 | 待采集 |
| write | hot | 4306.3 | 89.4% | 4398.1 | 91.4% | 1.021 | 待采集 | 待采集 |
| **best** | **cold** | **4046.6** | **84.1%** | **4111.1** | **85.4%** | **1.016** | 待采集 | 待采集 |
| **best** | **hot** | **4306.3** | **89.4%** | **4398.1** | **91.4%** | **1.021** | 待采集 | 待采集 |

### h20 / h200 对比分析

这组对比的价值在于：两卡同为 `sm_90a`、同为 141GB HBM3e、标称带宽同为 4814 GB/s，唯一
显著差异是 SM 数（78 vs 132，1.69×）与算力（BF16 148 vs 989 TFLOPS，6.7×）。带宽这个
常见的混淆变量被自然排除，于是「多出来的算力/占用率对访存上限有没有用」可以被单独观察。

- **h200 在三个 variant 上全面略高，但幅度分层明显**：read +7.3～7.8% > copy +2.9～3.3% >
  write +1.6～2.1%。SM 翻倍只在 read 上兑现得多。
- **解释**：write / copy 已经贴近同一块 HBM 的物理墙，再多的 SM 也搬不快；read 是三者中
  达成率最低的（cold 68.9% / 73.9%），瓶颈不在 HBM，而在读出依赖链与 in-flight 请求数，
  更多 SM 与更高占用率能直接补上这块延迟。
- **对优化的含义**：凡是靠「更多并行掩盖访存延迟」的手段（更宽访存、更多在飞 load、
  提高占用率），h20 的 headroom 大于 h200；而纯粹提高 HBM 效率的手段在两卡上收益接近。
  这正对应「同一优化在不同机器上收益可能反号」的机制。
- **cold vs hot**：两卡 hot 都稳定高于 cold（h20 best +259.7 GB/s / +6.4%，h200 best
  +287.1 GB/s / +7.0%），这部分是启动开销与冷缓存；单次启动场景要按 cold 报。
- **单次 HBM 流量上限**：h20 约标称 84%（cold）–89%（hot），h200 约 85%–91%，两卡差距在
  read 上最大、write 上最小。

### a100 占位

a100 是三机论证的第三点，用来说明「算力弱 ≠ 免受带宽限制」：其 BF16 ridge point（153）
反而高于 h20（31），因为它的带宽（2.039 TB/s）掉得比算力更快。缺这一台，「h20 是带宽
超配、算力阉割的特例」这句话就还没有对照证据。

- **状态**：待采集。所有依赖 a100 数据描述与对比的段落，文章里一律以「（a100 待采集）」占位，
  不得用外推数字填充。
- **2026-09-15 尝试作废（数字不予采用）**：目标机 8 张 A100 全部被另一份 8 卡训练作业占用
  （`utilization.gpu` 100%），逐 kernel 埋事件可见周期性 ~2.8 ms 抢占停顿，批量口径带宽被
  压到 ~943 GB/s（真实约 ~1690 GB/s）。按独占性判据（实测前 util 必须为 0）本次无效：
  未写入本目录 CSV，未回填 `bench/machine-peaks.json`。
- **待办**：等目标卡 `utilization.gpu == 0` 且可锁频后重跑 `bench-probe`，随后一次性回填
  本表 a100 两列、`bench/machine-peaks.json` 的 `a100.bw_ceiling_*`、以及
  `docs/environment-matrix.md` 的 a100 实测行。设备参数与作废原因见该文件 a100 一节。

### 文章论证骨架（供 content repo 撰写引用）

本仓库只提供事实与骨架，正本在 `miraclefarms-content`。本篇的论证顺序：

1. 为什么先钉基线：算力:带宽比（ridge point）决定一个优化该不该做。
2. 三机 ridge point：h20 vs h200 是受控对照（6.6× 完全由算力产生）；a100 作反例（占位）。
3. 测量口径：同一条测量路径、cold/hot 分开、分位数而非均值。
4. h20 / h200 实测对比：带宽差异分层（read > copy > write）及其占用率解释。
5. a100 位置与待采集状态（占位）。
6. 结论与失效边界：当前基线仅支持同会话相对比值，绝对上限待锁频重采。

## 截图

![cold latency](../../figures/01-execution-model/fig-01-execution-model-cold-latency.png)
![cold bandwidth](../../figures/01-execution-model/fig-01-execution-model-cold-bandwidth.png)
![hot latency](../../figures/01-execution-model/fig-01-execution-model-hot-latency.png)
![hot bandwidth](../../figures/01-execution-model/fig-01-execution-model-hot-bandwidth.png)

## 复现

```bash
cmake --build build --target bench-probe -j
./build/kernels/01-execution-model/bench-probe
```
