# 03-smem-staging

搬运范式的分水岭：`cp.async` 与它的天花板。把 tile 从 global 喂进计算，四种写法，
看瓶颈到底在「搬多少」还是「谁在搬、搬的时候谁在等」。

## 要验证的问题

内存层级的瓶颈不在容量，而在搬运期间线程能不能干别的。具体拆成三个可量化的问题：

1. **SMEM 值不值得搬**：把 tile 抄进 SMEM（`sync-stage`）比直接在 global 上反复读（`global`）快吗？
   tile 多大时开始划算？
2. **换指令还是换用法**：`cp.async` 的收益有多少来自指令本身（`cp-async-wait`），有多少来自
   「搬运与计算重叠」（`cp-async`）？
3. **失效点在哪**：tile 形状、访问模式（1d 连续 / 2d 分行）变化时，`cp.async` 在哪里不再赢，
   甚至变慢？

**预期在哪台机器不同**：`cp.async` 从 sm_80 起可用，本篇是 A100 真正参与论证的一篇——三台机器
跑的是同一条指令。A100 的 SMEM 上限更小（见下），64 元素的 tile 在 A100 上不存在；h20 算力受限、
h200 算力富余，重叠计算的收益取决于「计算在不在等搬运」，两台的答案可能相反。

以上都是待验证的假设。冒烟运行里已经看到重叠未必带来收益，采集后以数据为准。

## 四个变体

| 变体 | 角色 | 做法 | SMEM |
|---|---|---|---|
| `global` | 对照 | 不搬，依赖链直接在 global 上跑 passes 趟，L2 兜底 | 0 |
| `sync-stage` | **baseline** | `ld.global` + `st.shared` 逐元素抄进 SMEM，再算 | 1 槽 |
| `cp-async-wait` | 拆分对照 | 发 `cp.async` 搬 tile，立刻 `wait_all`，再算 | 1 槽 |
| `cp-async` | **optimized** | 发下一个 tile 的 `cp.async`，趁它在飞算当前 tile，算完再 `wait_all` | 2 槽 |

工作负载：一块 `bench::kStreamBufferBytes`（256 MiB）的 float 数组，切成 Rows × Cols 的 tile，
每个 tile 跑 `passes` 趟依赖链 `s = 0.5*s + v[m]`，结果写回 global。依赖链让每一趟都得重读数据，
编译器合并不了。

## tile 形状与 SMEM 用量

SMEM 数组大小由 tile 形状决定，必须编译期确定，所以形状是固定的实例化表，运行时用 `--shapes` 选子集。

| tile | 模式 | 元素 | `sync-stage` / `cp-async-wait` SMEM | `cp-async` SMEM | sm_80 | sm_90a |
|---|---|---|---|---|---|---|
| 1×4 | 1d | 4 | 4 KiB | 8 KiB | ✅ | ✅ |
| 1×16 | 1d | 16 | 16 KiB | 32 KiB | ✅ | ✅ |
| 4×4 | 2d | 16 | 16 KiB | 32 KiB | ✅ | ✅ |
| 2×8 | 2d | 16 | 16 KiB | 32 KiB | ✅ | ✅ |
| 1×64 | 1d | 64 | 64 KiB | 128 KiB | ❌ | ✅ |
| 4×16 | 2d | 64 | 64 KiB | 128 KiB | ❌ | ✅ |
| 8×8 | 2d | 64 | 64 KiB | 128 KiB | ❌ | ✅ |

SMEM = 256 线程 × 元素数 × 4 B（`cp-async` 再乘 2 个槽）。

**A100 上 64 元素 tile 编译不过**，这是实测到的边界，不是配置问题：`ptxas` 对 sm_80 目标限制每个
入口函数的 SMEM 为 `0xc000`（48 KiB），64 KiB 的 `sync-stage` 就报 `uses too much shared data`；
sm_90a 目标能放下 128 KiB。`CMakeLists.txt` 在目标架构含 sm_8x 时不编这三个形状。这就是提纲
第 1 节「每层的容量与代价」里 SMEM 那一格的数据。

2d 模式的行宽固定为 4092：是 4 的倍数（`cp.async` 要求 16 字节对齐），不是 8 的倍数，所以列数
≥ 8 的 tile 在每行末尾都会被裁小。被裁小的边缘 tile 在所有 staging 变体里一律走同步路径。

## `cp.async` 的三个局限在代码里的位置

提纲第 6 节列出的局限，也是第 04 篇 TMA 的动机清单，都能在 `staging.cu` 里找到对应代码：

| 局限 | 代码位置 |
|---|---|
| 地址计算仍在 SM | `stage_cp_async`：每行行首地址 `(y0 + r) * width + x0` 由 SM 计算，每 16 字节一条指令 |
| 多维 tile 仍要手算 | 2d tile 拆成 Rows 组独立的行搬运，行数越多指令组越多 |
| 边界仍要手判 | `can_cp_async`：非完整 tile 或列数不是 4 的倍数时退回 `stage_sync` |

## 目录代码

| 文件 | 作用 |
|---|---|
| `cp_async.cuh` | **本篇零件：SMEM staging 模板**。`cp_async_16`（一条 `cp.async.cg.shared.global`）、`cp_async_row`、`cp_async_wait_all` |
| `staging.cu` | 四个变体的 kernel、形状分发、host 黄金结果与逐位校验、cold/hot 计时 |
| `CMakeLists.txt` | 产出目标 `kernel-03-smem-staging`；按目标架构决定是否编 64 元素 tile |

**为什么不用 `cuda::memcpy_async`**：在 sm_90 上，16 字节对齐的拷贝配 SMEM 里的 barrier，
`memcpy_async` 会自动改走 `cp.async.bulk`，那是第 04 篇（TMA）的内容；配非 SMEM barrier 时，
它又会在函数内部立刻 `cp.async.wait_all`，重叠就没了。本篇要在三台机器上比同一条指令，还要
自己控制等待的位置，所以直接写指令。写法与 CUDA 13.3.1 头文件里 libcudacxx 的
`__cp_async_shared_global<16>` 与 `memcpy_completion` 一致。

运行参数：`--shapes 1x4,4x4`（形状子集）、`--passes`（默认 4）、`--width`（2d 行宽，默认 4092，
必须是 4 的倍数）、`--samples`（默认 30）、`--warmup`（默认 5）、`--batch`（默认 10）。

## 所用技术

- `__shared__ __align__(128)` SMEM 数组，按 `threadIdx.x` 分行
- `cp.async.cg.shared.global`（sm_80+）：16 字节一条的 global → SMEM 异步搬运，跳过 L1
- `cp.async.wait_all`：只阻塞本线程，等本线程发出的全部 `cp.async` 写完
- grid 按 `multiProcessorCount * 32` 推导，tile 按 grid-stride 分配给线程
- 共用计时 `bench::measure_cold` / `measure_hot`，工作集 `bench::kStreamBufferBytes`
- 每个形状、每个变体先与 host 黄金结果逐位比对再计时（依赖链里 `0.5*s` 精确可表示，host 与
  device 同序计算应逐位一致）

## 实验数据

**待采集。** 三台机器均无有效数据。

编译状态：sm_90a 与 sm_80 在 CUDA 13.3.1 容器内均编译通过。2026-09-15 在 h20 上做过一次冒烟
运行（5 个样本、未提交的工作树）：7 个形状 × 4 个变体的逐位校验全部通过。该次数字不满足
`git_dirty=no`，**不落盘、不引用**。

采集时要回填：

- 每台机器、每个形状：四个变体相对 `sync-stage` 的 median 加速比（cold / hot 分开）
- `cp-async` 与 `cp-async-wait` 的差：重叠本身值多少
- 1d vs 2d 同元素数（1×16 vs 4×4 vs 2×8，1×64 vs 4×16 vs 8×8）：行数与边界对 `cp.async` 的影响
- A100 与 h20/h200 在同一形状上的结论是否反号

## 截图

待生成。数据采集后由下面的命令产出：

- `../../figures/03-smem-staging/fig-03-smem-staging-cold-speedup-vs-sync-stage.png`
- `../../figures/03-smem-staging/fig-03-smem-staging-hot-speedup-vs-sync-stage.png`
- `../../figures/03-smem-staging/fig-03-smem-staging-cold-sweep.png`

## 复现

```bash
cmake --build build --target kernel-03-smem-staging -j
python3 tools/gpu_quiet_gate.py --need 1 --csv results/03-smem-staging/$(date +%F)-h20.csv \
    -- python3 tools/run.py --kernel 03-smem-staging --machine h20
python3 tools/plot.py results/03-smem-staging/*.csv -o figures/03-smem-staging/ \
    --view sweep --metric speedup --baseline sync-stage
```

A100 需要单独用 `-DCMAKE_CUDA_ARCHITECTURES=80` 构建，且先按 `docs/environment-matrix.md` 导出
forward-compat 的 `LD_LIBRARY_PATH`。
