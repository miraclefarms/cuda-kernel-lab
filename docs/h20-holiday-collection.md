# 首三篇 H20 采集与验收计划

本文是 01–03 篇的执行交接，不记录 SSH 别名、容器名或远端路径。执行者进入已有的 CUDA
开发容器，同步**已提交**的代码后逐项采集；实验 CSV、配图与数据提交由执行者完成。

## 三篇分别证明什么

| 篇目 | H20 实验问题 | 必需证据 | 论证边界 |
|---|---|---|---|
| 01 | 这张 H20 的 streaming 上限与 SM 资源限制是什么？ | 锁频的 read/copy/write 冷热数据；实际寄存器数 × SMEM 的静态 occupancy 扫描 | occupancy API 给的是理论驻留量，不是实测性能或实测 occupancy |
| 02 | 同一个 kernel 换一种计时方法，结论会变多少？ | 7 档 L2 相对尺寸 × 5 种计时协议；逐样本数据；锁频与未锁频会话 | 三个错误协议只用于演示，不可挪作正式 benchmark |
| 03 | `cp.async` 在什么 tile 上获益、何时反而变慢？ | 7 个形状 × 4 个变体 × cold/hot；先校验再计时 | `cp.async` 是 Ampere 时代的指令，不能称为 CUDA 13 新特性 |

这一轮只要求 H20。9 月 14–15 日的 H20/H200 CSV 未锁频，作为探索性背景保留；新图不混入
旧数据，也不据单机实验声称跨机器收益反号。A100/H200 在以后另起一轮、各自扫描并验收。

## 采集前

1. 将包含本计划与脚本的**干净提交**同步到容器，记下 40 位 commit。采集期间不修改代码或
   文档；`results/`、`figures/` 下新生成的文件允许未跟踪。
2. 选一张可用的物理 GPU，以下用 `GPU=0` 示范。预检只阻塞目标卡，其他卡忙不影响这轮采集。
3. 在容器内的仓库根目录执行：

   ```bash
   GPU=0  # 按实际可用卡号修改
   ./tools/collect_h20.sh build
   ./tools/collect_h20.sh preflight "$GPU"
   ```

所有 `BLOCK` 必须先解决。保存 `--matrix` 输出，供 `docs/environment-matrix.md` 回填；确认卡型
是 H20-3e、MIG Disabled、所选时钟在支持列表中。预检需有 `nvcc`；没有 ncu 时，
计数器论断要降级，时延实验仍可做。

脚本默认只对目标卡锁 **1800 MHz**，离开时复位。要改用另一档支持的时钟，设置
`H20_SM_CLOCK_MHZ=<MHz>`。锁频失败会停止采集；`02-unlocked` 是故意不锁频的对照。

## 逐项执行

```bash
./tools/collect_h20.sh 01 "$GPU"
./tools/collect_h20.sh 02 "$GPU"
./tools/collect_h20.sh 02-unlocked "$GPU"
H20_RUN_TAG=unlocked2 ./tools/collect_h20.sh 02-unlocked "$GPU"
H20_RUN_TAG=locked2 ./tools/collect_h20.sh 02 "$GPU"
./tools/collect_h20.sh 03 "$GPU"
```

02 的锁频/未锁频按 **A–B–B–A** 顺序各跑两次，减小时间漂移与先后顺序的混淆。脚本在
目标卡连续 60 秒 `utilization.gpu=0` 后启动，结束后再检查 30 秒。每次产出主 CSV 与
`.quiet.json`；01 另产出 `results/01-execution-model/occupancy/<日期>-h20.csv`；02 另产出
`<日期>-h20-samples.csv`。静默窗口通过后自动运行校验器。POST 失败会创建 `<CSV>.invalid`，
主 CSV 与逐样本侧车文件都不可引用。

脚本不会覆盖同一天的输出。其他重跑须设置唯一的 `H20_RUN_TAG`，保留旧次并写明重跑原因。
每行 CSV 的 `git_commit` 才是实验对应的代码版本。静默门禁无法发现恰好在 RUN 期间开始又
结束的短时共租户；文章必须保留这条限制。

## 数据验收与分析

- 校验器通过；`git_dirty=no`、`mig=Disabled`；锁频轮 `clocks_locked=yes`，未锁频对照为
  `no`；主 CSV 与 `.quiet.json` 的机器、日期和 commit 对得上。
- 检查所有 shape，包括负结果。正式计时的 p90/p10 大于约 1.05 时先排查；它只是复核信号，
  不能据此删掉不喜欢的点。02 的错误协议出现长尾，本来就是要展示的现象。
- 01 看 occupancy CSV 的 `actual_regs_per_thread` 和 `local_bytes_per_thread`；
  `requested_accumulators` 只是生成压力候选。静态 occupancy 只能给驻留上界，不能证明
  “100% 最快”。
- 02 在同一张卡、同一尺寸比较锁频 A–B–B–A，逐样本计算 wallclock 的 mean/median/p90。
  不预设均值一定朝哪个方向偏。
- 03 冷热分开，分别报告相对 `sync-stage` 的收益；再比较 `cp-async` 和
  `cp-async-wait`，拆出重叠收益。保留 `global` 或同步版本更快的形状。
- 仅凭延迟不能断定 occupancy、sector 或指令数机制。要写这类因果解释时，再选正负各一两个
  代表形状采 Nsight Compute 计数器；计数器方案按第一轮曲线确定。

先用新 01 的 best cold/hot 更新 `bench/machine-peaks.json`，保证配图参考线来自同一受控会话。
出图时**显式指定本轮验收通过的 H20 CSV**，不要用会混入旧文件或未锁频对照的通配符：

```bash
python3 tools/plot.py results/01-execution-model/<日期>-h20.csv \
  -o figures/01-execution-model/
python3 tools/plot.py results/02-measurement-discipline/<日期>-h20.csv \
  -o figures/02-measurement-discipline/ --view sweep
python3 tools/plot.py results/03-smem-staging/<日期>-h20.csv \
  -o figures/03-smem-staging/ --view sweep --metric speedup --baseline sync-stage
```

将 `<日期>` 换成真实日期，若有 tag 则加上 tag。检查图之后，回填每篇 README 的实验表和截图、
`docs/environment-matrix.md`。代码与数据分开提交，每次公开提交前运行
`python3 tools/check_confidential.py --staged`。

配套内容仓库需要另外纠正旧提案的 H20 96GB/4.0 TB/s 口径，以实际的 141GB HBM3e/约
4.8 TB/s 为准；同步审核后的图，并填写文章 pin 的代码 commit、CSV、环境和状态。
本文所在仓库不存公众号稿。
