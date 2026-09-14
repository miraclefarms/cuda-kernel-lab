# cuda-kernel-lab

CUDA 13 语言特性与硬件优化系列的代码与实验仓库。

每一篇文章对应这里的一个目录：`baseline` 与 `optimized` 两份可编译代码、一份可复现的 benchmark、每台机器一份 CSV 原始数据，以及由数据直接生成的配图。**文章里出现的每个数字，这里都能跑出来。**

配套文章：https://miraclefarms.github.io

## 为什么是三台机器

系列的核心主张是「一个优化在什么条件下反而更慢」。要证明这件事，需要的不是一台最强的卡，而是**算力带宽比拉开差距的一组卡**：

| 代号 | GPU | 架构 | BF16 dense | 带宽 | Ridge point |
|------|-----|------|-----------|------|-------------|
| `a100` | A100 80GB | `sm_80` Ampere | 312 TFLOPS | 2.0 TB/s | ≈156 FLOP/B |
| `h20` | H20 96GB | `sm_90a` Hopper | 148 TFLOPS | 4.0 TB/s | **≈37 FLOP/B** |
| `h200` | H200 141GB | `sm_90a` Hopper | ≈989 TFLOPS | 4.8 TB/s | **≈206 FLOP/B** |

- `a100` 没有 TMA、cluster、`wgmma`、FP8，是「新硬件特性之前你必须怎么写」的对照组。
- `h20` 与 `h200` **同一套 ISA、同一份二进制**，算力差 6.7 倍，ridge point 差 5.6 倍。同一个优化在这两台上的收益可以反号——这是本仓库最有价值的实验条件。

> 本仓库目前没有 Blackwell / Rubin 硬件。涉及 tcgen05、TMEM、CTA pair、NVFP4 的内容在文章里明确标注为「未实测·规格推演」，本仓库不提供对应代码。

## 篇目索引

| # | 主题 | 代码 | 文章 |
|---|------|------|------|
| 01 | 执行模型与三机基线画像 | `bench/src/probe_main.cu` | 待发布 |
| 02 | 内存层级与搬运范式 | — | 待发布 |
| 03 | 可信 microbenchmark 方法论 | `bench/` | 待发布 |
| 04–08 | 异步与流水线（TMA / mbarrier / warp specialization / cluster / PDL）| — | 待发布 |
| 09–13 | 张量核心（`mma.sync` → `wgmma` → FP8 → epilogue → 代际推演）| — | 待发布 |
| 14–17 | 编程模型上移（CuTe / CUTLASS / cuTile / 四路对照）| — | 待发布 |
| 18–19 | Capstone | `capstone/` | 待发布 |

## 快速开始

需要 CUDA Toolkit 13.x 与一块 Ampere 或更新的 GPU。

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a   # H20 / H200；A100 用 80
cmake --build build -j

./build/bench/bench-probe                        # 打印本机基线画像
python3 tools/run.py --kernel 00-template --machine h200
python3 tools/plot.py results/00-template/*.csv -o figures/00-template/
```

## 测量口径

所有数字都遵守同一套纪律，细节见 [docs/measurement-methodology.md](docs/measurement-methodology.md)：

- **cold 与 hot 分开报**：cold 每次 L2 flush + 单次启动，hot 在 CUDA Graph 内重放
- **只报分位数**（min / p10 / median / p90），不报均值
- **没锁频不报绝对 TFLOPS**，改用同一次会话内的相对比值
- 每条记录都带环境元数据（驱动、toolkit、实际时钟、MIG 状态、是否独占）

如果你在自己的机器上跑出了不同的结论，那大概率是真的不同——请开 issue 附上 CSV，这正是这个仓库想收集的东西。

## 文档

- [docs/measurement-methodology.md](docs/measurement-methodology.md) — 测量方法与 CSV schema
- [docs/interface-contract.md](docs/interface-contract.md) — 跨篇复用的四项接口约定
- [docs/environment-matrix.md](docs/environment-matrix.md) — 三台机器的环境记录
- [docs/reproduce.md](docs/reproduce.md) — 复现步骤与常见坑

## License

MIT
