# 测量方法与 CSV schema

这个文件定义整个仓库的数字口径。改动它等于改动所有已发表数字的含义，必须同步更新
`bench/` 并在文章里说明。

## 两种口径，永远分开报

| 口径 | 做法 | 回答什么问题 |
|------|------|-------------|
| `cold` | 每个采样前 flush L2，只计时一次 kernel 启动 | 一层只跑一次时的真实延迟 |
| `hot` | `batch` 次启动录进 CUDA Graph，重放后除以 batch | decode 循环里连续重放的吞吐 |

带宽受限 kernel 上两者能差几倍。任何把它们混在一张图里的呈现都是错的。

## 为什么必须 flush L2

`cudaDeviceSynchronize` 不清 L2。重复读同一块输入的 kernel，不 flush 测到的是 L2 带宽，
报出的 HBM 数字会高出数倍。`bench::L2Flusher` 按 `l2CacheSize × 2` 分配并在采样间写穿它。

## 为什么只报分位数

远程机器多半不独占、多半锁不了频。均值会把这些噪声悄悄折进头条数字，分位数不会。
统一报 min / p10 / median / p90 / max，图上用 p10–p90 作误差棒。

## 锁频与绝对值

`nvidia-smi -lgc` 需要 root。拿得到就锁，`tools/run.py --clocks-locked` 记录这一事实。
**拿不到就不报绝对 TFLOPS**，结论改用同一次会话内的相对比值（optimized / baseline），
并在文章里写明这一口径。`clocks_locked` 列存在的意义就是让读者能判断该信多少。

## ncu 计数器

采计数器默认需要 `NVreg_RestrictProfilingToAdminUsers=0` 或 root。用
`tools/env_capture.sh` 先确认。拿不到权限时，退到 wall-clock + 带宽利用率推断，
并在文章里显式声明「本篇无 ncu 计数器」，不要用别处的计数器数字冒充。

## CSV schema

测量列由 kernel 二进制打到 stdout，环境列由 `tools/run.py` 在运行时采集后前置。
二进制不报告它无法核实的环境——手填的环境等于不可复现。

**环境列（run.py 采集）**

`timestamp, machine, gpu_name, driver, sm_clock_mhz, mem_clock_mhz, compute_mode, ecc, mig, clocks_locked, toolkit, git_commit, git_dirty`

**测量列（二进制输出）**

`kernel, variant, mode, shape, dtype, bytes, flops, samples, min_ms, p10_ms, median_ms, p90_ms, max_ms`

**派生列（run.py 计算）**

`achieved_gbps, achieved_tflops` —— 均按 median 计算。

约定：
- `machine` 用 GPU 型号代号 `a100` / `h20` / `h200`。**型号必须写准**——它是查
  `docs/environment-matrix.md` 基线表的键，达成率的分母由它决定。不写主机名、IP、
  集群路径、用户名；同型号的不同机器不做区分
- `bytes` 是一次调用搬运的总字节数（读 + 写），不适用时填 0
- `git_dirty=yes` 的数据不得进文章。文章引用的必须是干净树上的 commit

## 达成率的分母

带宽利用率、算力利用率一律以 `docs/environment-matrix.md` 第一部分的**厂商标称 dense
峰值**为分母，不用实测峰值，也不用稀疏数字。三台机器共用同一套口径，跨机比较才成立。

## 文件命名

`results/{NN}-{slug}/{YYYY-MM-DD}-{machine}.csv`，每台机器一份，不合并。
合并是出图脚本的事，不是采集的事。

## 负结果

跑不出预期收益就照实记。这个系列的核心资产是失效边界，负结果与正结果同等重要。
不调参凑结论，不悄悄换 shape 直到数字好看——换了 shape 就把两组 shape 都报出来。
