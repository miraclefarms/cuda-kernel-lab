# 环境矩阵与对比基线

本文件有两部分，作用完全不同：

- **第一部分：厂商标称基线**——公开 datasheet 数字，三台卡共用同一套口径，**是所有文章
  计算「达成率」的分母**。这部分是固定的，不随机器变化。
- **第二部分：本机实测环境**——每台机器真实的驱动、toolkit、时钟、权限状态，由
  `tools/preflight.sh --matrix` 采集后填入。这部分决定某篇文章的数字能说到多硬。

## 第一部分：厂商标称基线（公开 datasheet）

三张卡全部按 **SXM / dense（非 sparsity）** 口径记录。NVIDIA 官方常同时给出 2:4 结构化
稀疏的数字（正好是 dense 的两倍），本项目一律用 dense——稀疏数字对手写 kernel 没有意义。

| | **a100** | **h20** | **h200** |
|---|---|---|---|
| 全称 | A100 80GB SXM4 | H20 141GB HBM3e SXM | H200 141GB SXM |
| 架构 | Ampere | Hopper | Hopper |
| Compute capability | `sm_80` | `sm_90a` | `sm_90a` |
| SM 数 | 108 | 78 | 132 |
| 显存 | 80 GB HBM2e | 141 GB HBM3e | 141 GB HBM3e |
| **显存带宽** | **2.039 TB/s** | **4.8 TB/s** | **4.8 TB/s** |
| FP64 (vector) | 9.7 TFLOPS | **1 TFLOPS** | 34 TFLOPS |
| FP64 Tensor | 19.5 TFLOPS | — | 67 TFLOPS |
| FP32 | 19.5 TFLOPS | 44 TFLOPS | 67 TFLOPS |
| TF32 Tensor | 156 TFLOPS | 74 TFLOPS | 495 TFLOPS |
| **BF16/FP16 Tensor** | **312 TFLOPS** | **148 TFLOPS** | **989 TFLOPS** |
| **FP8 Tensor** | 不支持 | **296 TFLOPS** | **1979 TFLOPS** |
| TDP | 400 W | 500 W | 700 W |

### 由基线导出的三个关键比值

Ridge point = 峰值算力 ÷ 峰值带宽，单位 FLOP/byte。一个 kernel 的算术强度低于它就是
带宽受限，高于它就是算力受限。**这是本系列判断「一个优化该不该做」的第一把尺子。**

| Ridge point | a100 | h20 | h200 |
|---|---|---|---|
| BF16 | 153 | **31** | 206 |
| FP8 | — | 62 | 412 |

- **h20 与 h200 构成一个近乎受控的实验。** 两者同为 `sm_90a`、同为 141GB HBM3e、
  理论带宽几乎相同（4814 vs 4800 GB/s），**唯一的显著差异是算力：148 vs 989 TFLOPS，
  6.7 倍**。因此 ridge point 的 6.6 倍差距完全由算力一侧产生，带宽这个混淆变量被自然
  排除掉了。同一份二进制在两台机器上，一台该做的是喂满 4.8 TB/s，另一台该做的是把
  989 TFLOPS 填满——「同一个优化收益反号」的论证条件比原先设想的 96GB 版本更干净。
- **a100 的 ridge point 反而比 h20 高。** 算力弱不等于更容易受带宽限制——h20 是
  「带宽超配、算力阉割」的特例，这一点在国内实际部署里非常常见，但几乎没有公开数据。
- **h20 的 FP64 只有 1 TFLOPS**，是 h200 的 1/34、a100 的 1/10。CUDA 13.4 起 cuBLAS
  能在 Ampere 及以后用 Ozaki-II 方案以张量核心仿真 FP64——在 h20 上这个仿真极可能
  大幅超过原生 FP64。这是本项目硬件条件下一个几乎无人做过的实测机会。

### 特性可用性

| | a100 | h20 | h200 |
|---|---|---|---|
| `cp.async` | ✅ | ✅ | ✅ |
| `mma.sync` | ✅ | ✅ | ✅ |
| TMA / `cp.async.bulk` | ❌ | ✅ | ✅ |
| Thread block cluster / DSMEM | ❌ | ✅ | ✅ |
| `wgmma` | ❌ | ✅ | ✅ |
| FP8 tensor core | ❌ | ✅ | ✅ |
| PDL / `setmaxnreg` | ❌ | ✅ | ✅ |
| `tcgen05` / TMEM / NVFP4 | ❌ | ❌ | ❌ |
| CUDA Tile（13.2 起下探 `sm_8x`）| ✅ | ✅ | ✅ |

### 来源

- A100 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- H200 datasheet — https://www.nvidia.com/en-us/data-center/h200/
- H20 参数为公开报道与经销商 spec sheet 汇总，NVIDIA 未发布面向全球的正式 datasheet。
  本系列实测为 **141GB HBM3e 版本**：`bench-probe` 报 memory bus 6016 bit、
  memory clock 3201 MHz，导出理论带宽 4814 GB/s，与 4.8 TB/s 标称一致；SM 78、
  L2 60 MiB。96GB HBM3 版本（4.0 TB/s）不用于本系列。

> L2 容量、实际 boost 时钟、SMEM/SM 等由 `bench-probe` 在真机上读取，不在此表预填——
> 猜一个数字进基线表，比留空危险得多。

---

## 第二部分：本机实测环境

每台机器第一次跑实验前用 `tools/preflight.sh --matrix` 填一次，驱动或 toolkit 变更后重填。
只记型号与版本，不记主机名、IP、集群路径、用户名。

### a100 — A100 80GB SXM4 (`sm_80`)

| 项 | 值 |
|----|-----|
| 驱动版本 | 待填 |
| CUDA Toolkit | 待填 |
| 容器镜像 | 待填 |
| bench-probe 输出 | 待填（L2、SMEM/SM、实际时钟）|
| 实测 HBM 带宽 / 标称 | 待填 |
| 锁频权限 | 待确认 |
| ncu 计数器权限 | 待确认 |
| MIG | 待确认 |
| 独占 | 待确认 |

### h20 — H20 141GB HBM3e SXM (`sm_90a`)

| 项 | 值 |
|----|-----|
| 驱动版本 | 590.44.01 |
| CUDA Toolkit | 13.3.73（V13.3.73）|
| 容器镜像 | 未使用（裸机，无 docker）|
| bench-probe 输出 | cc 9.0；SM 78；L2 60 MiB；显存 139.8 GiB；memory bus 6016 bit；memory clock 3201 MHz；SM clock 1980 MHz；smem/SM 228 KiB；默认功耗上限 500 W |
| 实测 streaming 上限 / 标称 | ⚠ **需重测**。旧值 read 3852 / copy 3870 / write 4012 GB/s （标称 4814，80–83%）来自 `bench-probe` 自建的计时循环，未 flush L2 且未走 `measure_cold/hot`，被 `00-template` 的 4113 GB/s 超出 6%，不能当分母。`bench-probe` 已改为与 kernel 同一套纪律，重跑后回填此处与 `bench/machine-peaks.json` |
| 锁频权限 | ✅ root 可锁（`nvidia-smi -lgc` 实测成功，测完已 `-rgc`）|
| ncu 计数器权限 | ✅ root 可采（`ncu` 2026.2.1 实测通过）|
| MIG | Disabled（8 卡均未切分）|
| 独占 | 8 卡整机，型号 `H20-3e`；跑分时须确认无其他进程占用目标卡 |

> 本机为 **141GB HBM3e 版本**（带宽 4.8 TB/s），不是 96GB HBM3（4.0 TB/s）。
> 基线表与 ridge point 已按此口径记录。8 张卡同型号，数据列一律记 `h20`。

### h200 — H200 141GB SXM (`sm_90a`)

| 项 | 值 |
|----|-----|
| 驱动版本 | 待填 |
| CUDA Toolkit | 待填 |
| 容器镜像 | 待填 |
| bench-probe 输出 | 待填 |
| 实测 HBM 带宽 / 标称 | 待填 |
| 锁频权限 | 待确认 |
| ncu 计数器权限 | 待确认 |
| MIG | 待确认 |
| 独占 | 待确认 |

---

## 四个阻塞项

任何一条不成立，第 03 篇的测量口径就得改。**开工第一件事就是验证它们。**

h20 上的实测状态（2026-09-14）：

1. **驱动版本 ≥ 580.65.06** — ✅ 590.44.01。
2. **ncu 计数器权限** — ✅ root 下 `ncu` 2026.2.1 可采。
3. **锁频权限** — ✅ root 下 `nvidia-smi -lgc` 可锁，故可报绝对 TFLOPS。
4. **MIG 与独占状态** — MIG Disabled；8 卡整机但非独占保证，跑分前需确认目标卡空闲。

1. **驱动版本 ≥ 580.65.06** — CUDA 13.x 的最低驱动要求。远程集群常年跑 535/550/570，
   驱动不够则本项目的容器起不来，CUDA 13 特性全部不可用。这是唯一一个可能直接否掉
   整个系列技术选型的前提，必须最先确认。
2. **ncu 计数器权限** — 默认 `NVreg_RestrictProfilingToAdminUsers=1` 时非 root 采不到。
   拿不到就只能 wall-clock + 带宽利用率推断，文章需显式声明无计数器证据。
3. **锁频权限** — `nvidia-smi -lgc` 需 root。拿不到就不报绝对 TFLOPS，改用会话内相对比值。
4. **MIG 与独占状态** — MIG 实例下没有完整 SM 视图，cluster 行为受限，数据不可比。
