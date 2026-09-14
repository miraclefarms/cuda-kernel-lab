---
name: kernel-debug
description: cuda-kernel-lab 的实验排查手册——数字可疑、两次跑结果不一致、达成率异常高或异常低、优化没有收益或收益大到不合理时，按固定顺序定位原因。触发：「这个数字不对」「怎么比理论峰值还高」「两次跑差很多」「优化没效果」「带宽只有 X%」「结果不可复现」。
---

# 数字可疑时的排查顺序

按这个顺序查，不要跳步——列在前面的原因出现频率高出一个量级。

**先明确一件事：跑不出预期收益本身是合法结果。** 这个系列的核心资产是失效边界。排查是为了排除「测错了」，不是为了把数字调好看。排除完仍然没有收益，就如实记录负结果，标明复现条件。

## 1. 正确性

最贵的错误是快的错 kernel。先确认 optimized 的输出对齐 baseline，再看任何时间。

边界条件、尾块处理、`n` 不能整除 tile 时的分支——这些地方出错往往正好让 kernel 少做事，于是"变快了"。

## 2. 口径是否混了

- cold 和 hot 拿去对比了吗？两者在小 kernel 上能差几倍
- CSV 里 `bytes` / `flops` 填对了吗？triad 是 3 份流量（2 读 1 写），copy 是 2 份
- 达成率的分母用对了吗？理论峰值和 streaming 天花板是两个数，见 `docs/measurement-methodology.md`

## 3. L2

**达成率超过 80% 就要警惕，超过 100% 一定是这里出了问题。**

```bash
./build/bench/bench-probe | grep -i "L2"
```

工作集必须远大于 L2（h20 是 60 MiB）。用 `bench::kStreamBufferBytes` 就不会错。自己写计时循环而绕过 `measure_cold` 的话，L2 不会被 flush——`bench-probe` 早期版本正是这样报出过一个比真实 kernel 还低的天花板。

## 4. 机器状态

```bash
./tools/preflight.sh
grep -E "sm_clock|clocks_locked|git_dirty|mig" results/{NN}-{slug}/*.csv
```

- 两次跑差 20% 以上：多半没锁频或机器不独占。看 CSV 的 `sm_clock_mhz` 是否一致
- p10–p90 离散大：同上，结论改用同一次会话内的相对比值
- `git_dirty=yes`：这份数据不能进文章，提交干净后重跑

## 5. 网格与占用率

```bash
./build/bench/bench-probe    # SM 数、寄存器/SM、SMEM/SM
```

- 网格是不是写死的常数？三台机器 108 / 78 / 132 SM，常数网格在其中两台上必然错
- 寄存器溢出？编译时加 `--ptxas-options=-v` 看 spill
- 流水级数吃掉的 SMEM 是否把驻留块数压到 1

## 6. 编译

- `-DCMAKE_CUDA_ARCHITECTURES=90a` 的 `a` 有没有省掉？省掉后 `wgmma`/TMA/cluster 静默不可用
- 是不是 Debug 构建？本仓库默认 Release，确认没有被覆盖
- 交叉编译只能证明「编得过」，不能替代在目标机器上真跑

## 7. ncu

拿得到计数器权限时（`preflight.sh` 会说），先看这三项：

```bash
ncu --set full -o prof ./build/kernels/{NN}-{slug}/kernel-{NN}-{slug}
```

- **内存吞吐 vs 计算吞吐**：确认这个 kernel 到底受限于哪一侧，再决定优化方向
- **warp stall reason**：等内存、等 barrier、还是等指令发射
- **occupancy achieved vs theoretical**：差距大说明有尾效应或负载不均

拿不到权限时，退到 wall-clock + 带宽利用率推断，并在文章里声明本篇无计数器证据。

## 8. 仍然解释不了

记录成负结果：贴出配置、CSV、preflight 输出，说明尝试过什么、排除了什么。**不要反复调参直到数字好看**——那样得到的是一个不可复现的巧合，而不是结论。
