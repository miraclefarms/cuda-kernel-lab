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

- **h20 与 h200 提供同 ISA、近似同带宽的对照。** 两者同为 `sm_90a`、同为 141GB HBM3e，
  理论带宽接近（4814 vs 4800 GB/s），但 SM 数、算力及其他实现细节并不相同。
  BF16 ridge point 的 6.6 倍差距主要来自标称算力差（148 vs 989 TFLOPS）；
  具体 kernel 的瓶颈仍需结合算术强度和计数器判断，收益反号须靠同题扫描实证。
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
| 驱动版本 | 575.51.03（**低于 CUDA 13 的 580.65.06**）|
| CUDA Toolkit | 13.3.73（V13.3.73）|
| CUDA UMD（forward-compat）| **580.178.04**，`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`（见下）|
| 容器镜像 | cuda-kernel-lab 复现容器（`nvidia/cuda:13.3.1-devel-ubuntu24.04`）|
| bench-probe 输出 | SM 108；cc 8.0；L2 40 MiB；smem/SM 164 KiB；smem/block opt-in 163 KiB；memory bus 5120 bit；mem clock 1593 MHz；theoretical 2039 GB/s（**仅供参照，不作基线**）|
| 实测 HBM 带宽 / 标称 | **作废，不予采用（2026-09-15 非独占）** |
| 锁频权限 | ✅ root 可锁（`nvidia-smi -lgc` 可用；本次未锁，因数据作废）|
| ncu 计数器权限 | ✅ ncu 2026.2.1 存在（本次未采）|
| MIG | Disabled |
| 独占 | ❌ 目标机 8 卡被另一份训练作业占用，`utilization.gpu` 100% |

> **2026-09-15 a100 采集尝试作废——数字不予采用、未落盘。** 8 张 A100 全部被另一份 8 卡
> 训练作业占用（每卡 ~77 GB、`utilization.gpu` 100%、380–440 W）。逐 kernel 埋事件显示每
> ~12.5 个 kernel 就有一次 ~2.8 ms 的周期性抢占停顿，hot 口径带宽从 ~1690 GB/s 被压到
> ~943 GB/s，cold 也被压到理论值的 83%（干净 A100 通常 ~90%）。按
> `docs/measurement-methodology.md` 的独占性判据（util 必须为 0）本次无效：未写入
> `results/`，未回填 `bench/machine-peaks.json`（a100 的 ceiling 仍为 `null`）。
>
> **驱动口径（已定：走 forward-compat UMD）**：内核驱动 575.51.03 低于 CUDA 13 最低要求，
> 直接跑 CUDA 13.3 二进制报 `error 35`（insufficient driver）。只升 UMD 的前向兼容可行，
> **采用 `cuda-13.0` compat UMD 580.178.04**，运行前
> `export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`，不需要动宿主机内核模块。
> 2026-09-15 复测确认（`bench-probe` 打印 `driver version 13000` / `runtime version 13030`）：
> `cuda-13.1/13.2/13.3` 的 compat（UMD 590.48.01 / 595.91.07 / 610.43.02）配不上 575 内核模块，
> `cuInit` 全部返回 **803**（unsupported display driver / cuda driver combination）；只有 13.0 通过。
> 探测逻辑收敛在 `tools/cuda_compat.py`，`preflight.sh` 据此把 driver<580 降为 WARN，
> `tools/run.py` 每份 CSV 记录实际使用的 compat 目录（`cuda_compat` 列）。宿主机若日后升到
> ≥ 580，此列自动变空，无需改脚本。

### h20 — H20 141GB HBM3e SXM (`sm_90a`)

本机为 **141GB HBM3e 版本**（带宽 4.8 TB/s），不是 96GB HBM3（4.0 TB/s）。基线表与 ridge
point 已按此口径记录。8 张卡同型号，数据列一律记 `h20`。

#### 2026-09-26 锁频会话（首三篇 H20 采集，可引用）

| 项 | 值 |
|----|-----|
| 驱动版本 | 615.71.09（KMD = UMD = 615.71.09）|
| CUDA Toolkit | 13.3, V13.3.73 |
| CUDA UMD | 随内核驱动（native，无需 forward-compat）|
| 容器镜像 | `nvidia/cuda:13.3.1-devel-ubuntu24.04` 系 |
| bench-probe 输出 | cc 9.0；SM 78；max threads/SM 2048；regs/SM 65536；L2 60 MiB；显存 139.84 GiB；memory bus 6016 bit；memory clock 3201 MHz；SM clock 上限 1980 MHz；smem/SM 228 KiB；smem/block opt-in 227 KiB；theoretical 4814.30 GB/s；ECC on；async engine 3 |
| 实测 streaming 上限 / 标称 | best cold **3797.5** / best hot **4047.9** GB/s（read 3293.5 / 3949.3，copy 3755.8 / 3920.7，write 3797.5 / 4047.9，格式 cold / hot，median）；标称 4814 → 78.9% / 84.1%。锁频 **1800 MHz**、静默门禁（PRE 60 s / POST 30 s，`util=0`）通过、`git_dirty=no`。见 `results/01-execution-model/2026-09-26-h20.csv`（commit `0025113`）|
| 锁频权限 | ✅ root 可锁（`nvidia-smi -lgc` 1800,1800 成功，测完 `-rgc` 复位）|
| ncu 计数器权限 | ✅ root 可采（`ncu` 2026.2.1；03 篇定向采了 2 个形状）|
| MIG | Disabled（8 卡均未切分）|
| 独占 | 目标卡 `utilization.gpu` 门禁通过（PRE/POST 均 0）；残留风险：RUN 期间起止全在窗口内的短命共租户观测不到 |

> **同轮还采集了**：02 篇 7 档尺寸 × 5 协议（`2026-09-26-h20.csv` 等四份锁频/未锁频配对，
> commit `0025113`）、03 篇 7 形状 × 4 变体（`2026-09-26-h20.csv`，同 commit）。
> **环境坑（已修）**：该镜像的 `ld.so.conf` 把 `/usr/local/cuda-13.3/compat` 排在 native
> 驱动目录之前，导致 ≥580 的 KMD 载入旧 compat UMD（610.43.02）而 `cuInit` 报 803；
> `tools/cuda_compat.py` 现会探测并把可用的 native 目录前插（提交 `9d92b1b`）。
> `collect_h20.sh` 另修了 01 篇漏传 `bench-probe --csv` 的问题（提交 `0025113`）。

#### 2026-09-14 探索性会话（未锁频，非独占；只作相对比值）

| 项 | 值 |
|----|-----|
| 驱动版本 | 590.44.01 |
| CUDA Toolkit | 13.3.73（V13.3.73）|
| 容器镜像 | 未使用（裸机，无 docker）|
| bench-probe 输出 | cc 9.0；SM 78；max threads/SM 2048；regs/SM 65536；L2 60 MiB；显存 139.8 GiB；memory bus 6016 bit；memory clock 3201 MHz；SM clock 1980 MHz；smem/SM 228 KiB；默认功耗上限 500 W |
| 实测 streaming 上限 / 标称 | best cold **4046.6** / best hot **4306.3** GB/s（read 3317.6 / 3954.2，copy 3775.7 / 3928.8，write 4046.6 / 4306.3，格式 cold / hot）；标称 4814 → 84.1% / 89.5%。2026-09-14 采于**非独占**机器（8×vLLM worker 常驻）且**未锁频**，按纪律只作同会话相对比值。见 `results/01-execution-model/2026-09-14-h20.csv`（commit `dfa8153`）|
| 锁频权限 | ✅ root 可锁（实测成功，测完已 `-rgc`）|
| ncu 计数器权限 | ✅ root 可采（`ncu` 2026.2.1 实测通过）|
| MIG | Disabled（8 卡均未切分）|
| 独占 | ❌ 非独占（常驻 worker，仅偶发空窗）|

> 两轮差值主要来自**锁频**（1800 vs 1980 MHz）与**独占性**：锁频会话达成率更低但条件受控，
> `bench/machine-peaks.json` 的 H20 `bw_ceiling_*` 取 2026-09-26 锁频轮。

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

h20 上的实测状态（2026-09-26 锁频会话）：

1. **驱动版本 ≥ 580.65.06** — ✅ 615.71.09。
2. **ncu 计数器权限** — ✅ root 下 `ncu` 2026.2.1 可采（03 篇已用）。
3. **锁频权限** — ✅ root 下 `nvidia-smi -lgc` 可锁（本轮锁 1800 MHz），故可报绝对带宽。
4. **MIG 与独占状态** — MIG Disabled；8 卡整机，采集前以目标卡 `utilization.gpu == 0` 判定独占。

1. **驱动版本 ≥ 580.65.06** — CUDA 13.x 的最低驱动要求。远程集群常年跑 535/550/570，
   驱动不够则本项目的容器起不来，CUDA 13 特性全部不可用。这是唯一一个可能直接否掉
   整个系列技术选型的前提，必须最先确认。
2. **ncu 计数器权限** — 默认 `NVreg_RestrictProfilingToAdminUsers=1` 时非 root 采不到。
   拿不到就只能 wall-clock + 带宽利用率推断，文章需显式声明无计数器证据。
3. **锁频权限** — `nvidia-smi -lgc` 需 root。拿不到就不报绝对 TFLOPS，改用会话内相对比值。
4. **MIG 与独占状态** — MIG 实例下没有完整 SM 视图，cluster 行为受限，数据不可比。
