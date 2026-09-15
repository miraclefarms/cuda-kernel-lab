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

远程机器多半锁不了频。均值会把噪声悄悄折进头条数字，分位数不会。
统一报 min / p10 / median / p90 / max，图上用 p10–p90 作误差棒。

## 独占性判据：看利用率，不看进程数

**测量前目标卡的 `utilization.gpu` 必须为 0。** 允许存在进程——常驻的推理服务、
持有一个空闲 CUDA context 的 worker 都可以留着，只要它们没有在用 GPU。判定只看利用率，
不看 `nvidia-smi` 的进程数。

判据这样定的原因：GPU 在多个进程之间按时间片轮转，正在用 GPU 的进程会把我们的 kernel
周期性打断。本项目实测过一次反例——同卡另一份作业跑在 100% util 时，每 ~2ms 就插进一次
~2.8ms 的抢占停顿，批量（hot）口径的带宽被压到真实值的一半，而 `nvidia-smi` 的进程列表
甚至因为容器 PID 命名空间而看不到那个进程。**进程数不可靠，利用率才可靠。**

`tools/preflight.sh` 据此判定：任一可见 GPU 的 util > 0 → BLOCK，禁止采集；进程列表只作
信息输出，不参与判定。不满足就换卡或等空，**不存在「只报分位数」的折中**。

## 静默窗口门禁：tools/gpu_quiet_gate.py

`preflight.sh` 只做一次瞬时判定，等不到空窗就当场退出。跑一份长数据时用门禁脚本把任务
包起来，它等一段可验证的空闲窗口才放行，任务结束后再验证一次：

```bash
python3 tools/gpu_quiet_gate.py --need 1 --interval 10 --quiet-window 60 \
    --post-window 30 --csv results/{NN}-{slug}/{date}-{machine}.csv \
    --report /tmp/{NN}-{slug}-{machine}.quiet.json \
    -- python3 tools/run.py --kernel {NN}-{slug} --machine {machine}
```

- **默认 interval 10s、PRE 窗口 60s、POST 窗口 30s**。POST 不需要整分钟，2–3 个采样足以
  确认任务结束后没有残留共占。
- 判据按**时间跨度**而非样本数：最后一段连续 `util == 0` 覆盖满窗口才算数，避免采样抖动
  把 50s 当成 60s。
- 只要求 `--need` 张卡同时空闲（本系列 = 1）；选中后自动设 `CUDA_VISIBLE_DEVICES`，任务
  保证落在选中的卡上。显存占用不管，只盯 `utilization.gpu`。
- 退出码：`0` 通过 / `1` 任务自身失败 / `2` POST 检出共占（已跑完，按下文作废；传 `--csv`
  时自动建 `.invalid`）/ `3` PRE 等不到空窗 / `4` 环境错误（`nvidia-smi` 读不到、`[N/A]`、
  候选卡不存在）。`--self-test` 可无 GPU 验证窗口逻辑。
- 全程 stdout 打一行 JSON 判决，`--report` 另写完整采样摘要。

**残余风险，必须随数据声明**：RUN 期间不采样，一个「恰好在任务窗口内起、任务结束前停」
的短命共租户观测不到。PRE+POST 双向静默只能挡住持续型共占——a100 事件里的训练作业正是
这一类。这条写进报告的 `caveat` 字段，不得据此宣称绝对保证。

## 数据作废的标注

一次运行被判定不可用（非独占、工具链错配、事后发现污染等）时，**不要删掉假装没发生，
也不要留一个看起来可用的文件**。按落盘与否分两种处理：

- **未落盘**：在 `docs/environment-matrix.md` 对应机器一节记一条「作废」——日期、机器、
  原因、结论（数字不予采用）。本次 a100 尝试即属此类。
- **已落盘**：在该 CSV 所在目录加一个同名 `.invalid` 标记文件，并在篇目 README 写明原因；
  不要把它回填进 `bench/machine-peaks.json`。

作废的数据不得回填 `machine-peaks.json`，不得引用进文章，也不得当作「相对比值」的底数。

## 锁频与绝对值

`nvidia-smi -lgc` 需要 root。拿得到就锁，`tools/run.py --clocks-locked` 记录这一事实。
**拿不到就不报绝对 TFLOPS**，结论改用同一次会话内的相对比值（optimized / baseline），
并在文章里写明这一口径。`clocks_locked` 列存在的意义就是让读者能判断该信多少。

## 驱动版本与 forward-compat UMD

CUDA 13.x 名义上要求内核驱动 ≥ 580。拿不到 root 升不了驱动时，可以只升用户态驱动（UMD）：
镜像自带 `/usr/local/cuda-13.x/compat/`，把 `LD_LIBRARY_PATH` 指向本机内核模块配得上的
那个 compat 目录即可。哪个能用是实测事实——a100 那台 575 内核模块只认 `cuda-13.0`
（UMD 580.178.04），13.1/13.2/13.3 一律 `cuInit=803`。探测收敛在 `tools/cuda_compat.py`，
`preflight.sh` 据此把 driver<580 判为 WARN（而非 BLOCK），并把 UMD 写进
`docs/environment-matrix.md`。

`tools/run.py` 每份 CSV 记 `cuda_compat` 列 = 本次运行实际生效的 compat 目录（未用则空）。
这列存在的意义：读者看到 `driver=575` + `toolkit=13.3` 还能跑时，能知道靠的是哪条
forward-compat 路径，而不是怀疑数据来源。**用了 compat 的会话，绝对 TFLOPS 的解读要更保守**
——UMD 是较旧驱动栈的兼容层，口径与原生 ≥580 不同。

## ncu 计数器

采计数器默认需要 `NVreg_RestrictProfilingToAdminUsers=0` 或 root。用
`tools/preflight.sh` 先确认。拿不到权限时，退到 wall-clock + 带宽利用率推断，
并在文章里显式声明「本篇无 ncu 计数器」，不要用别处的计数器数字冒充。

## CSV schema

测量列由 kernel 二进制打到 stdout，环境列由 `tools/run.py` 在运行时采集后前置。
二进制不报告它无法核实的环境——手填的环境等于不可复现。

**环境列（run.py 采集）**

`timestamp, machine, gpu_name, driver, cuda_compat, sm_clock_mhz, mem_clock_mhz, compute_mode, ecc, mig, clocks_locked, toolkit, git_commit, git_dirty`

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

## 达成率的分母：两个都报

带宽类结论有两个合理分母，各自回答不同的问题，**同时报出来**：

- **理论峰值**（`environment-matrix.md` 第一部分，厂商标称 dense，不用稀疏数字）——
  回答「距离硬件上限还有多远」
- **实测 streaming 上限**（`bench-probe` 的 read / copy / write）——回答「距离一个写得
  很好的简单 kernel 还有多远」

两者差距本身就是信息：h20 上 `bench-probe` 的 streaming 只能到理论值的 80–83%，
说明即使完美的访问模式也拿不到标称带宽。只报理论值会让所有 kernel 看起来都很差；只报
实测上限会掩盖访问模式之外的损失。

算力类结论用理论 dense 峰值即可，没有等价的「实测上限」参照。

天花板与 kernel 必须用**同一套纪律和同一个工作集**测出来，否则不能相除：
`bench-probe` 走的是和所有 kernel 相同的 `measure_cold` / `measure_hot`，工作集统一为
`bench::kStreamBufferBytes`（256 MiB／缓冲区）。早期版本它自建计时循环、不 flush L2，
报出的「天花板」被 `00-template` 超出 6% —— 那不是天花板，是一个无关的数字。

分母写在 `bench/machine-peaks.json`，`tools/plot.py` 直接读它画参考线，所以图和文档
不可能对不上。某台机器的 `bw_ceiling_*_gbps` 为 `null` 表示还没在那台机器上跑过
`bench-probe`。

## 文件命名

`results/{NN}-{slug}/{YYYY-MM-DD}-{machine}.csv`，每台机器一份，不合并。
合并是出图脚本的事，不是采集的事。

## 负结果

跑不出预期收益就照实记。这个系列的核心资产是失效边界，负结果与正结果同等重要。
不调参凑结论，不悄悄换 shape 直到数字好看——换了 shape 就把两组 shape 都报出来。
