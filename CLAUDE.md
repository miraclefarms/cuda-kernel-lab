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

1. **代码**：`kernels/{NN}-{slug}/` 下 `baseline` 与 `optimized` 并列，可编译可运行
2. **数据**：`results/{NN}-{slug}/{date}-{machine}.csv`，schema 见 `docs/measurement-methodology.md`，**每台跑过的机器一份**
3. **配图**：`figures/{NN}-{slug}/*.png`，由 `tools/plot.py` 从 CSV 生成，**不手画**
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

同样禁止进入本仓库：机器主机名、IP、用户名、SSH 配置、集群路径、任何凭据。机器在数据里只用 `a100` / `h20` / `h200` 这种代号。

## 测量纪律

这些规则的存在理由是：远程机器多半拿不到 root，测量条件天然受限，所以口径必须写死。

- **没锁频就不报绝对 TFLOPS**。拿不到 `nvidia-smi -lgc` 权限时，结论一律用「同一次会话内的相对比值」表述，并在 CSV 里记录实际 SM 时钟。
- **一律报分位数**（min / p10 / median / p90），不报均值。机器不独占时尤其如此。
- **每次运行必须记录环境**：机器代号、GPU 型号、compute capability、驱动版本、CUDA toolkit 版本、SM/内存实际时钟、MIG 状态、是否独占、ECC 状态。`tools/env_capture.sh` 负责采集。
- **冷热两种口径分开报**：cold（每次 L2 flush + 单次启动）与 hot（CUDA Graph 内多次重放）。混在一起的数字没有意义。
- **跑不出预期收益就如实记录**。这个系列的核心资产是失效边界，负结果与正结果同等重要，不调参凑结论。

## 目录约定

```
bench/       harness：计时、L2 flush、设备探测、CSV 输出
kernels/     每篇一个目录 {NN}-{slug}/，baseline 与 optimized 并列
results/     CSV 原始数据，按 {NN}-{slug}/{date}-{machine}.csv
figures/     由 CSV 生成的 PNG
tools/       run.py / plot.py / env_capture.sh / check_confidential.py
docs/        接口约定、环境矩阵、测量方法论、复现说明
```

`{NN}` 是文章篇号（01–19），`{slug}` 是该篇的英文短名。目录名一旦建立不再改，因为文章里的 permalink 指向它。

## 接口约定

`docs/interface-contract.md` 定义四项跨篇接口：tile 描述符构造、pipeline 深度参数化、accumulator 布局、epilogue 回调签名。**这四项对齐了，S4 的 capstone 才拼得起来。** 改动它们需要同步检查所有已有 kernel。

## 常用命令

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a   # H20 / H200；A100 用 80
cmake --build build -j
./build/bench/bench-probe                       # 设备基线画像
python3 tools/run.py --kernel 00-template --machine h200
python3 tools/plot.py results/00-template/*.csv -o figures/00-template/
python3 tools/check_confidential.py
```

## 提交规范

- 直接提交推送 `main`，除非明确要求开分支
- commit message 用英文，格式 `{scope}: {summary}`，scope 取 `bench` / `kernels/NN-slug` / `tools` / `docs`
- 实验数据与代码分开提交，方便文章引用一个干净的代码 commit
