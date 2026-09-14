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

**待采集。** 需在 a100 / h20 / h200 三台各跑一次 `bench-probe`，把 best cold/hot 回填
`bench/machine-peaks.json` 与该机器在 `docs/environment-matrix.md` 第二部分的小节。

| 机器 | 实测 streaming 上限（cold / hot）| 标称带宽 | 状态 |
|---|---|---|---|
| a100 | — | 2039 GB/s | 待测 |
| h20 | — | 4814 GB/s | 需重测（旧值未 flush L2、口径不一致）|
| h200 | — | 4800 GB/s | 待测 |

## 截图

**待生成。** 三机跑完后按同一口径出「标称峰值 vs 实测 streaming 天花板」对比图，放入
`figures/01-execution-model/` 并在此嵌入。

## 复现

```bash
cmake --build build --target bench-probe -j
./build/kernels/01-execution-model/bench-probe
```
