# CLAUDE.md

本文件是 `cuda-kernel-lab` 仓库的工作约定。配套文章仓库是 `miraclefarms-content`（private）。

## 这是什么仓库

MiracleFarms「CUDA 13 语言特性与硬件优化」系列文章的**代码与实验仓库**。存放 kernel 代码、benchmark harness、实验原始数据和出图脚本。**public 仓库**，读者会 clone 它来复现文章里的每一个数字。

## 边界：这里不写文章（MANDATORY）

文章正本、公众号稿、配图、知识沉淀**全部在 `miraclefarms-content`**，不在这里。原因是硬依赖，不是习惯：

- 公开站点重建由 content repo 的 `_posts/** assets/** wiki-src/** _data/**` 变更触发
- 公众号流水线（`publish-wechat.js`、渲染主题、封面模板、发布记录、相关文章索引）全部是 content repo 本地的
- `update:source-coverage` / `validate:content` / `stats:knowledge` / `check:confidential` 扫的都是 content repo 路径
- 全部 skill 都挂在 content repo 的 `.agents/skills/`

**在本仓库的会话里不要做这些事**：写 `_posts` 风格的文章、生成公众号 markdown、调用 wechat/X 发布、追加 wiki Insight、维护知识索引。需要做这些，切到 `~/mycode/miraclefarms-content` 另开会话。

本仓库允许的产出：CUDA/C++ 代码、Python 工具、实验 CSV、图表 PNG、`docs/` 下的技术约定文档、README 索引表。

## 交接契约：一篇文章需要本仓库交付什么

每篇文章开写前，本仓库必须已经产出下面四项，缺一项就不该开写：

1. **代码**：`kernels/{NN}-{slug}/` 下 `baseline` 与 `optimized` 并列，可编译可运行（第 01 篇是基线画像，只有画像工具、无 optimized，见该目录 README），并附一份 `README.md` 覆盖代码、要测什么、所用技术、实验数据与截图
2. **数据**：`results/{NN}-{slug}/{date}-{machine}.csv`，schema 见 `docs/measurement-methodology.md`，**每台跑过的机器一份**
3. **配图**：`figures/{NN}-{slug}/*.png`，由 `tools/plot.py` 从 CSV 生成，**不手画**；样式真源 `tools/plot_style.py`（见 `figure-style` skill）
4. **pin 信息**：commit hash、机器列表、toolkit/驱动版本、CSV 路径——回填进 content repo 的 `private-workspace/projects/cuda-kernel-series/lab-pin.yaml`

配图同步到 content repo 的 `assets/{post-slug}/`；**CSV 留在本仓库**，private repo 不囤实验数据。

文章引用代码一律用 pinned permalink（40 位 commit hash），**不引用 `main`**。

## 保密规则（MANDATORY）

本仓库是 public，且**不在 content repo 的 `check:confidential` 扫描范围内**，所以这里自带一份检查：

```bash
python3 tools/check_confidential.py          # 全量
python3 tools/check_confidential.py --staged # 提交前
```

硬禁止：任何来自文章仓库 `private-workspace/` 的客户、合作方、厂商材料——具体型号、内部参数、路线图时间、封装与工艺选择、良率、软件栈内部实现。**加免责声明不能让保密内容变得可发布。**允许的只有带公开可引用来源的事实。

受限关键词清单放在 `tools/confidential-patterns.json`，**不入版本控制**（`tools/confidential-patterns.example.json` 是形状示例）。原因是：一份「什么名字绝对不能出现」的公开清单，本身就泄漏了存在哪些保密材料。

**明确允许**：NVIDIA 公开发布的产品型号及其官方标称参数——A100、H20、H100、H200、B200、GB200、RTX PRO 6000 等，以及它们 datasheet 上的算力、带宽、显存、SM 数、TDP。这些是公开可引用的事实，是本系列建立统一对比基线的前提，**必须写清楚**。含糊其辞地写「某张 Hopper 卡」反而让数据不可比。

禁止的是**机器身份**：主机名、IP、用户名、SSH 配置、集群路径、任何凭据。数据里的 `machine` 列用 `a100` / `h20` / `h200`——这是型号代号，不是机器代号，同型号的不同机器不做区分。

## 测量纪律

这些规则的存在理由是：远程机器多半拿不到 root，测量条件天然受限，所以口径必须写死。

- **测量前目标卡的 `utilization.gpu` 必须为 0**。**允许存在进程**（常驻推理服务、持有一个空闲 CUDA context 的 worker 都行），只要它没在用 GPU；判定只看利用率，不看进程数。有人在用 GPU 时按时间片轮转，会把我们的 kernel 周期性打断——实测同卡另一份作业跑到 100% util 时，hot 口径带宽被压到真实值的一半。这条不满足就不采，没有「只报分位数」的折中。
- **长数据用 `tools/gpu_quiet_gate.py` 包住**：目标卡连续空闲 60s 才启动任务，任务结束后再验 30s；POST 失败自动建 `.invalid`。用法、退出码与残余风险见 `docs/measurement-methodology.md`。
- **没锁频就不报绝对 TFLOPS**。拿不到 `nvidia-smi -lgc` 权限时，结论一律用「同一次会话内的相对比值」表述，并在 CSV 里记录实际 SM 时钟。
- **一律报分位数**（min / p10 / median / p90），不报均值。
- **每次运行必须记录环境**：机器代号、GPU 型号、compute capability、驱动版本、CUDA toolkit 版本、forward-compat UMD（未用则空）、SM/内存实际时钟、MIG 状态、是否独占、ECC 状态。`tools/run.py` 每次运行自动采集；换机器时先跑 `tools/preflight.sh`。
- **冷热两种口径分开报**：cold（每次 L2 flush + 单次启动）与 hot（CUDA Graph 内多次重放）。混在一起的数字没有意义。
- **跑不出预期收益就如实记录**。这个系列的核心资产是失效边界，负结果与正结果同等重要，不调参凑结论。

## 目录约定

```
bench/       共用 harness：计时、L2 flush、stream 上限、CSV schema。只放跨篇模块
kernels/     每篇一个目录 {NN}-{slug}/，baseline 与 optimized 并列
results/     CSV 原始数据，按 {NN}-{slug}/{date}-{machine}.csv
figures/     由 CSV 生成的 PNG
tools/       preflight.sh / run.py / plot.py / plot_style.py / check_confidential.py（共用）
.agents/skills/  kernel-experiment（数据生产闭环）、kernel-debug（排查手册）、figure-style（绘图规范）
docs/        接口约定、环境矩阵、测量方法论、容器、复现说明
docker/      复现容器：Dockerfile（digest 固定）+ run.sh
```

`{NN}` 是文章篇号（01–19），`{slug}` 是该篇的英文短名。目录名一旦建立不再改，因为文章里的 permalink 指向它。

### 代码归属（MANDATORY）

**一篇文章的专属代码必须能靠路径或文件名前缀认出属于哪一篇**，只有真正跨篇复用的模块例外：

- 单篇专属 → `kernels/{NN}-{slug}/`：该篇的可执行目标、源文件、README 都放这里
- 跨篇共用 → `bench/`（计时 / L2 flush / stream / CSV schema）、`tools/`（预检 / 运行 / 出图 / 保密检查）
- 判断标准：**删掉某一篇文章后它还该不该存在**。该留 → 共用；该删 → 属于那一篇
- 共用模块不得反向依赖单篇代码；单篇目录通过 `bench_headers` / `bench_stream` 链接共用模块
- 第 01 篇的 `bench-probe` 是专属工具（设备基线画像），放在 `kernels/01-execution-model/` 而不是 `bench/`；它回填的 `bench/machine-peaks.json` 是共用基线数据

### 篇目 README 约定（MANDATORY）

每个 `kernels/{NN}-{slug}/` 必须有一份 `README.md`，不必像文章那样面面俱到，但要覆盖：

- **要验证的问题**：这篇证明什么，预期在哪台机器收益反号
- **目录代码**：每个源文件 / target 的作用，baseline 与 optimized 分别在哪
- **所用技术**：涉及的 CUDA 13 / 硬件特性，一句话说明为什么用它
- **实验数据**：从 `results/{NN}-{slug}/*.csv` 抄关键行成表，写结论与失效边界；未采集时显式写「待采集」
- **截图**：嵌入 `figures/{NN}-{slug}/*.png`（相对路径 `../../figures/{NN}-{slug}/...`）
- **复现**：build / run / plot 三条命令

形状以 `kernels/00-template/README.md` 为准。数据与截图两节必须随实验回填，不能把「待采集」留到开写。

## 接口约定

`docs/interface-contract.md` 定义四项跨篇接口：tile 描述符构造、pipeline 深度参数化、accumulator 布局、epilogue 回调签名。**这四项对齐了，S4 的 capstone 才拼得起来。** 改动它们需要同步检查所有已有 kernel。

## 换机器时先跑预检

```bash
./tools/preflight.sh --matrix
```

退出码非 0 表示有 BLOCK 项（驱动 < 580、MIG 开启、缺 nvcc、**目标卡利用率不为 0**），
**在它通过之前不要采集任何数据**。WARN 项不阻塞，但决定这次数据能说什么：没锁频不报绝对
TFLOPS，没有 ncu 就不给计数器证据。进程数不是判据——常驻但空闲的进程允许存在，只有
「正在用 GPU」才禁止采集。

## Skill

- `kernel-experiment` —— 一篇文章的数据生产闭环（预检 → 建目录 → 写码 → 三机编译 → 扫参 → CSV → 图 → 回填 → 提交）
- `kernel-debug` —— 数字可疑时的固定排查顺序
- `figure-style` —— 统一绘图规范与配色模板；配图相关请求走这里（唯一真源 `tools/plot_style.py`）

## 常用命令

```bash
docker build -t cuda-kernel-lab:13.3.1 docker/  # 复现容器，见 docs/container.md
./docker/run.sh                                 # 进容器（宿主机驱动需 >= 580.65.06）

./tools/setup.sh                                # 进容器第一件事：装工具 + 建 build + 预检
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a   # H20 / H200；A100 用 80
cmake --build build -j
./build/kernels/01-execution-model/bench-probe  # 设备基线画像
python3 tools/run.py --kernel 00-template --machine h200
python3 tools/plot.py results/00-template/*.csv -o figures/00-template/  # 延迟 + 带宽两种视图
python3 tools/check_confidential.py
```

## 提交规范

- 直接提交推送 `main`，除非明确要求开分支
- commit message 用英文，格式 `{scope}: {summary}`，scope 取 `bench` / `kernels/NN-slug` / `tools` / `docs`
- 实验数据与代码分开提交，方便文章引用一个干净的代码 commit
