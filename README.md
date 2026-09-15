# cuda-kernel-lab

CUDA 13 语言特性与硬件优化系列的代码与实验仓库。

每一篇文章对应这里的一个目录：`baseline` 与 `optimized` 两份可编译代码、一份可复现的 benchmark、每台机器一份 CSV 原始数据，以及由数据直接生成的配图。**文章里出现的每个数字，这里都能跑出来。**

配套文章：https://miraclefarms.github.io

## 为什么是三台机器

系列的核心主张是「一个优化在什么条件下反而更慢」。要证明这件事，需要的不是一台最强的卡，而是**算力带宽比拉开差距的一组卡**：

| 代号 | GPU | 架构 | SM | BF16 dense | 带宽 | Ridge point |
|------|-----|------|----|-----------|------|-------------|
| `a100` | A100 80GB SXM4 | `sm_80` Ampere | 108 | 312 TFLOPS | 2.039 TB/s | 153 FLOP/B |
| `h20` | H20 141GB HBM3e SXM | `sm_90a` Hopper | 78 | 148 TFLOPS | 4.8 TB/s | **31 FLOP/B** |
| `h200` | H200 141GB SXM | `sm_90a` Hopper | 132 | 989 TFLOPS | 4.8 TB/s | **206 FLOP/B** |

全部按 SXM / dense（非 2:4 稀疏）口径。完整基线表、特性可用性矩阵与来源见
[docs/environment-matrix.md](docs/environment-matrix.md)。

- `a100` 没有 TMA、cluster、`wgmma`、FP8，是「新硬件特性之前你必须怎么写」的对照组。
- `h20` 与 `h200` **同一套 ISA、同一份二进制**，算力差 6.7 倍，ridge point 差 6.6 倍。同一个优化在这两台上的收益可以反号——这是本仓库最有价值的实验条件。
- `a100` 的 ridge point（153）反而比 `h20`（31）高。算力弱不等于更容易撞带宽——`h20` 是「带宽超配、算力阉割」的特例，而这恰恰是国内大量实际部署所用的卡。

> 本仓库目前没有 Blackwell / Rubin 硬件。涉及 tcgen05、TMEM、CTA pair、NVFP4 的内容在文章里明确标注为「未实测·规格推演」，本仓库不提供对应代码。

## 篇目索引

| # | 主题 | 代码 | 文章 |
|---|------|------|------|
| 01 | 执行模型与三机基线画像 | `kernels/01-execution-model/` | 待发布 |
| 02 | 内存层级与搬运范式 | — | 待发布 |
| 03 | 可信 microbenchmark 方法论 | `bench/` | 待发布 |
| 04–08 | 异步与流水线（TMA / mbarrier / warp specialization / cluster / PDL）| — | 待发布 |
| 09–13 | 张量核心（`mma.sync` → `wgmma` → FP8 → epilogue → 代际推演）| — | 待发布 |
| 14–17 | 编程模型上移（CuTe / CUTLASS / cuTile / 四路对照）| — | 待发布 |
| 18–19 | Capstone | `capstone/` | 待发布 |

每篇的主张、实验设计与零件见 [docs/series-outline.md](docs/series-outline.md)。篇序仍有未决项（02/03 顺序），以该文件「未决事项」为准。

## 快速开始

推荐走容器：宿主机只需要 NVIDIA 驱动（**≥ 580.65.06**，CUDA 13.x 的最低要求）和
NVIDIA Container Toolkit，其余全在镜像里。

```bash
docker build -t cuda-kernel-lab:13.3.1 docker/
./docker/run.sh bash -lc '
  ./tools/setup.sh &&                                # 装工具 + 按 GPU 建 build + 预检
  ./build/kernels/01-execution-model/bench-probe &&
  python3 tools/run.py --kernel 00-template --machine h200 &&
  python3 tools/plot.py results/00-template/*.csv -o figures/00-template/
'
```

镜像按 digest 固定在 `nvidia/cuda:13.3.1-devel-ubuntu24.04`，另装了 Nsight Compute、
CMake 与出图用的 matplotlib。细节与 ncu 权限说明见 [docs/container.md](docs/container.md)。

不用容器也可以，本机装 CUDA Toolkit 13.x + CMake ≥ 3.24 + Python 3.10 后直接跑同样的
cmake / python 命令即可。

## 测量口径

所有数字都遵守同一套纪律，细节见 [docs/measurement-methodology.md](docs/measurement-methodology.md)：

- **cold 与 hot 分开报**：cold 每次 L2 flush + 单次启动，hot 在 CUDA Graph 内重放
- **只报分位数**（min / p10 / median / p90），不报均值
- **没锁频不报绝对 TFLOPS**，改用同一次会话内的相对比值
- 每条记录都带环境元数据（驱动、toolkit、实际时钟、MIG 状态、是否独占）

如果你在自己的机器上跑出了不同的结论，那大概率是真的不同——请开 issue 附上 CSV，这正是这个仓库想收集的东西。

## 文档

- [docs/series-outline.md](docs/series-outline.md) — 全系列 19 篇提纲：每篇的主张、实验、零件与建议目录
- [docs/measurement-methodology.md](docs/measurement-methodology.md) — 测量方法与 CSV schema
- [docs/interface-contract.md](docs/interface-contract.md) — 跨篇复用的四项接口约定
- [docs/environment-matrix.md](docs/environment-matrix.md) — 三台机器的环境记录
- [docs/container.md](docs/container.md) — 复现容器与宿主机前提
- [docs/reproduce.md](docs/reproduce.md) — 复现步骤与常见坑

## License

MIT
