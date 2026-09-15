---
name: kernel-experiment
description: 在 cuda-kernel-lab 里为某一篇文章生产代码与实验数据的完整闭环——机器预检、建篇目目录、写 baseline/optimized、三机编译、扫参数、跑 CSV、出图、回填基线与 pin。触发：「新开一篇 kernel」「写第 N 篇的代码」「在这台机器上跑数据」「收集 h200/a100 数据」「补一组扫描」，或任何要在本仓库产出 kernel 代码或实验数据的请求。
---

# 一篇文章的数据生产闭环

把一次调用当作完整任务：代码能跑、数据可信、图能用、回填做完，才算结束。

## 读取入口

1. 本仓库 `CLAUDE.md`（边界与纪律）
2. `docs/measurement-methodology.md`（口径与 CSV schema）
3. `bench/machine-peaks.json`（达成率分母）
4. `docs/interface-contract.md`（四项跨篇接口）

## 第 0 步：预检，不可跳过

```bash
./tools/preflight.sh --matrix
```

- 退出码非 0 → **停止**。BLOCK 项（驱动 < 580、MIG 开启、无 nvcc、**目标卡 `utilization.gpu` 不为 0**）会让数据不可用或不可比。
- WARN 项不阻塞，但**决定这次数据能说什么**：没锁频就不能报绝对 TFLOPS，没有 ncu 就不能给计数器证据。把这些约束记下来，写文章时逐条声明。
- WARN `driver < 580` 走的是 forward-compat UMD：preflight 会打印要导出的 `LD_LIBRARY_PATH`（探测见 `tools/cuda_compat.py`），**跑数据前必须 export，否则报 `error 35`**；`run.py` 会把实际用的 compat 目录记进 CSV 的 `cuda_compat` 列。
- 独占判据看**利用率不看进程数**：常驻但空闲的进程（推理服务、空闲 context）允许存在；只有「正在用 GPU」才禁止采集。跨容器 PID 命名空间时进程列表可能看不到占用者，别信进程数。
- 第一次上某台机器时，用 `--matrix` 的输出回填 `docs/environment-matrix.md` 第二部分。

## 机器分工（决定在哪台机器做什么）

| 机器 | 角色 | 说明 |
|------|------|------|
| `h20` | 开发与正确性主力 | `sm_90a`，写码、调试、验证正确性都在这里 |
| `h200` | 性能扫描与上限验证 | 同为 `sm_90a`，**同一份二进制直接跑**，不重复开发 |
| `a100` | 对照组 | `sm_80`，只在「没有这个特性时怎么写」构成论点的篇目跑 |

**三条硬规则**，违反会让整个系列的核心论点失效：

1. **禁止把某台机器的最优值写死。** h20 的 ridge point 是 31，h200 是 206；在 h20 上「够用」的 tile 形状和流水深度，到 h200 上可能远远喂不饱算力。所有形状 / 深度 / 缓冲级数必须是可扫描的旋钮（模板参数或运行时参数），不是常数。
2. **网格尺寸一律从 device props 推导**，不写常数。三台机器 SM 数是 108 / 78 / 132，差异很大。参照 `kernels/00-template` 里的 `multiProcessorCount * 32`。
3. **性能结论必须每台机器各自扫描**，禁止跨机复用最优配置。交付判据是「本机正确 + 三机都能编译」。

## 新开一篇

```bash
cp -r kernels/00-template kernels/{NN}-{slug}
```

`{NN}` 是篇号（01–19），`{slug}` 是英文短名。**目录名建立后不再改**——文章里的 permalink 指向它。在根 `CMakeLists.txt` 加一行 `add_subdirectory(kernels/{NN}-{slug})`，并改子目录 CMake 的 target 名。

写一份 `README.md`（不必像文章，但必须覆盖，形状以 `kernels/00-template/README.md` 为准）：

- **要验证的问题**：这篇证明什么，预期在哪台机器收益反号
- **目录代码**：每个源文件 / target 的作用，baseline 与 optimized 分别在哪
- **所用技术**：涉及的 CUDA 13 / 硬件特性，一句话说明为什么用
- **实验数据**：从 `results/{NN}-{slug}/*.csv` 抄关键行成表，写结论与失效边界；未采集时显式写「待采集」
- **截图**：嵌入 `figures/{NN}-{slug}/*.png`（相对路径 `../../figures/{NN}-{slug}/...`）
- **复现**：build / run / plot 三条命令

数据跑完后回来把「实验数据」与「截图」两节补上，不能把「待采集」留到开写。

## 写代码的硬约束

- **单篇代码只放 `kernels/{NN}-{slug}/`**，不要塞进 `bench/`；只有真正跨篇复用的模块才进 `bench/`（判断标准：删掉这篇它还该不该存在）
- **`baseline` 与 `optimized` 并列**，跑同一问题、同一输入
- **正确性先于性能**：optimized 的输出先对齐 baseline 再计时，快的错 kernel 是最贵的错误
- **cold 与 hot 都输出**，走 `bench::measure_cold` / `measure_hot`，不要自己写计时循环——`bench-probe` 曾经因为自建计时而报出一个比真实 kernel 还低的「天花板」
- **带宽类 kernel 的工作集用 `bench::kStreamBufferBytes`**，这样它和 `bench-probe` 的天花板才可比
- CSV 打到 stdout，schema 见 `bench/include/bench/csv.hpp`，`bytes` 填一次调用的 HBM 流量（读+写），不适用填 0

## 编译与交叉验证

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a && cmake --build build -j   # 本机（H 系）
cmake -B build-sm80 -DCMAKE_CUDA_ARCHITECTURES=80 && cmake --build build-sm80 -j   # 交叉验证能编过
```

架构必须显式指定，`90a` 的 `a` 不能省（`wgmma`、TMA、cluster 都是 arch-conditional）。A100 路径只能在 H 系上验证「编得过」，**正确性与性能必须在 A100 上真跑**。

## 跑数据

```bash
sudo nvidia-smi -lgc <min>,<max>          # 拿得到 root 时才做
python3 tools/run.py --kernel {NN}-{slug} --machine h20 --clocks-locked -- --n ... 
sudo nvidia-smi -rgc                       # 测完立刻复位
```

- `--clocks-locked` **只有真锁了才传**，它会写进 CSV 决定读者信多少
- **长数据用门禁包起来**：`tools/gpu_quiet_gate.py` 等目标卡连续空闲 60s 才启动任务，任务结束后再验 30s，POST 失败自动建 `.invalid`。用法与退出码见 `docs/measurement-methodology.md`
- 开跑前再确认目标卡 `utilization.gpu == 0`；采集途中若有人在用，这份数据作废，按 `docs/measurement-methodology.md` 标注，不要落盘假装可用
- 跑完检查 CSV：`git_dirty` 必须是 `no`，`mig` 必须 `Disabled`，p10–p90 离散过大说明机器不干净
- 扫参数时每个配置一行，不要只留最优的那行——收益曲线和失效边界都在被扔掉的那些行里

## 出图

```bash
python3 tools/plot.py results/{NN}-{slug}/*.csv -o figures/{NN}-{slug}/
```

带宽受限的篇目看 `-bandwidth.png`（带理论峰值与 streaming 天花板参考线），算力受限看 `-latency.png`。图一律由 CSV 生成，**不手画**。

配色与样式的唯一真源是 `tools/plot_style.py`（Nature/Science 期刊风 + Okabe-Ito 色盲安全调色板 + 斜纹冗余编码），完整规范见 `figure-style` skill。**不要在任何 kernel 目录里自建颜色或 matplotlib 样式**；新增图类型也要 import `plot_style`。改了样式必须重跑所有已存在的图。

## 回填（四处，缺一不可）

1. **`kernels/{NN}-{slug}/README.md`** 的「实验数据」与「截图」两节：从 CSV 抄关键行成表，嵌入生成的图
2. **`bench/machine-peaks.json`**：首次在某机器跑 `bench-probe` 后，把 best cold/hot 写进 `bw_ceiling_*_gbps`
3. **`docs/environment-matrix.md`** 第二部分：驱动、toolkit、容器、探测输出、权限状态
4. **content repo 的 `lab-pin.yaml`**：commit hash、机器、版本、CSV 路径、已同步配图

## 提交

代码与数据**分开提交**，文章要引用的是一个干净的代码 commit：

```bash
git add kernels/{NN}-{slug} bench && git commit -m "kernels/{NN}-{slug}: <what it measures>"
git add results figures && git commit -m "results/{NN}-{slug}: <machine> data"
python3 tools/check_confidential.py --staged
```

## 完成判据

- [ ] preflight 通过，WARN 项已记录成文章要声明的口径限制
- [ ] baseline 与 optimized 都在，正确性已验证
- [ ] 本机 + 交叉编译都通过
- [ ] CSV 齐全，`git_dirty=no`，扫描的全部配置都保留
- [ ] 图已生成
- [ ] 篇目 README 的实验数据与截图两节已回填，无「待采集」
- [ ] 四处回填完成
- [ ] 保密闸门通过
