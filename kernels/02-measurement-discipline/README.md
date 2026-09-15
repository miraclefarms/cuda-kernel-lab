# 02-measurement-discipline

一个可信的数字需要多少纪律。kernel 固定不变（00-template 的 float4 triad），只换计时方法，
看同一个 kernel 能被量出多少个「结果」。

## 要验证的问题

大量 CUDA 性能结论不可复现，原因集中在四件事。这篇把每一件都做成一个可以单独量化的变体：

| 错误 | 对应变体 | 它错在哪 |
|---|---|---|
| 没 flush L2 | `no-flush` | 采样之间不清 L2。工作集装得进 L2 时，第二次起读的是 L2 不是 HBM，报出来的是 L2 带宽 |
| 混了冷热口径 | `batched` | 一个计时区间里连跑 20 次再除以 20，却当成「单次启动延迟」报。启动开销被摊薄，冷缓存只在第一次出现。本仓库 `bench-probe` 的第一版就这么测，报出的「天花板」被 00-template 超过 6% |
| 没预热、host 计时 | `wallclock` | host 时钟包住「启动 + 同步」，不预热。host 侧开销与 OS 抖动全算进去，形成长尾 |
| 报了均值 | 原始样本（`--samples-out`）| 长尾会被均值折进头条数字，分位数不会 |

正确的测法是 `disciplined`，直接调用 `bench::measure_cold` / `measure_hot`，全仓库其余篇目都走它。

**扫描维度是缓冲区大小**，按本机 L2 容量推导（L2 的 1/16 到 2 倍），最后一档是全仓库统一的
工作集 `bench::kStreamBufferBytes`（256 MiB）。shape 列里的 `ws_over_l2` 是三块缓冲区之和除以 L2：

- `ws_over_l2 < 1`：工作集装得进 L2，预期 `no-flush` / `batched` 与 `disciplined` 差得最多
- `ws_over_l2 ≫ 1`：L2 只能装下一小部分，flush 与否的差距应当收敛

**预期在哪台机器不同**：三台卡 L2 不同（a100 40 MiB、h20 60 MiB），同一个绝对尺寸在一台上
装得进、另一台装不进。所以横轴用 `ws_over_l2` 对齐，不用 KiB 对齐。

以上是待验证的假设，不是结论。尤其是「不 flush 能测出高于 HBM 标称的带宽」：小尺寸下单次
启动的固定开销也在变大，两者谁占上风要看数据。测不出来就照实写。

### cold 和 hot 是什么，为什么两个都要

这两个词在本系列所有篇目里含义固定：

- **cold（冷启动）**：每次计时前先把 L2 清空，然后只启动**一次** kernel，计这一次的耗时。
  它回答「一个 kernel 在一次前向里只跑一遍时，实际要花多久」——比如模型每层各执行一次的算子。
  这个数字里包括启动开销和冷缓存。
- **hot（热重放）**：把连续 20 次启动录成一个 CUDA Graph，整体重放，耗时除以 20。
  它回答「同一个 kernel 在循环里反复跑，稳定下来以后每次多快」——比如 decode 循环。
  启动开销被摊薄，缓存已经是热的。

两个数都对，只是回答的问题不同，所以永远分开报，也只和同口径的数比（cold 比 cold，hot 比 hot）。
`batched` 这个错误变体，就是把 hot 的做法冒充成 cold 报出来。

## 目录代码

| 文件 | 作用 |
|---|---|
| `discipline.cu` | 一个 kernel（`triad_vectorized`）+ 四种计时协议。每个尺寸先校验结果再计时；可选 `--samples-out` 输出逐样本原始耗时 |
| `CMakeLists.txt` | 产出目标 `kernel-02-measurement-discipline` |

这篇的 baseline / optimized 与其他篇目的意思不同：kernel 没有两份，**错误的测法是 baseline，
`bench/` 的测法是 optimized**。三个错误协议是故意写错的，只用于演示，不要拿去测别的 kernel。

运行参数：`--sizes-kib a,b,c`（覆盖默认尺寸表）、`--samples`（默认 50）、`--warmup`（默认 5）、
`--batch`（默认 20）、`--samples-out <path>`。

## 所用技术

- `bench::L2Flusher`：按 `l2CacheSize × 2` 分配缓冲区，采样之间写穿它，逼出 L2 里的旧数据
- `cudaEvent` 事件对计时（`no-flush` / `batched` / `disciplined`）与 host `std::chrono`（`wallclock`）对照
- CUDA Graph 重放（`measure_hot`）：把 launch 开销从被测对象里摘出去
- 分位数统计（`bench::summarize`）；本篇给 `measure_cold` / `measure_hot` 加了可选的原始样本输出，统计口径不变
- 尺寸从 `cudaDeviceProp::l2CacheSize` 推导，不写常数

## 本篇在代码之外的部分

提纲第 4、5 节不是这个二进制能单独回答的，需要按下面的方式采集：

- **锁频 vs 不锁频**（第 4 节）：同一台机器跑两次，一次 `sudo nvidia-smi -lgc` 锁频后带
  `--clocks-locked`，一次不锁频带 `--tag unlocked`。两份 CSV 的 `sm_clock_mhz` 与分位数宽度直接对比
- **ncu 计数器**（第 5 节）：待补。h20 上 root 可采（`docs/environment-matrix.md`），采集命令与
  结论口径在跑数据时补进本节

## 实验数据

**待采集。** 三台机器均无有效数据。

2026-09-15 在 h20 上做过一次冒烟运行（10 个样本、未提交的工作树）：全部 7 个尺寸的正确性
校验通过，五个变体都能跑完。该次数字不满足 `git_dirty=no`，**不落盘、不引用**。

采集时要回填：

- 每台机器一张表：各尺寸 × 各变体的 median 带宽，以及相对 `disciplined` 的偏差
- `wallclock` 与 `disciplined` 的 mean / median / p90 对照（来自 samples 文件）
- `no-flush` / `batched` 相对 `disciplined` 的偏差随 `ws_over_l2` 的变化，在哪一档收敛
- 锁频前后同一变体的 p10–p90 宽度

## 截图

待生成。数据采集后由下面的命令产出：

- `../../figures/02-measurement-discipline/fig-02-measurement-discipline-cold-sweep.png`
- `../../figures/02-measurement-discipline/fig-02-measurement-discipline-hot-sweep.png`
- `../../figures/02-measurement-discipline/fig-02-measurement-discipline-cold-speedup-vs-disciplined.png`

## 复现

```bash
cmake --build build --target kernel-02-measurement-discipline -j
python3 tools/gpu_quiet_gate.py --need 1 --csv results/02-measurement-discipline/$(date +%F)-h20.csv \
    -- python3 tools/run.py --kernel 02-measurement-discipline --machine h20 --clocks-locked \
    -- --samples-out results/02-measurement-discipline/$(date +%F)-h20-samples.csv
python3 tools/plot.py results/02-measurement-discipline/*-h20.csv results/02-measurement-discipline/*-h200.csv \
    -o figures/02-measurement-discipline/ --view sweep
```

`--clocks-locked` 只有真的执行过 `nvidia-smi -lgc` 才能传。samples 文件没有环境列，靠文件名与
同日同机器的主 CSV 对应。
