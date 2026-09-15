---
name: figure-style
description: cuda-kernel-lab 的统一绘图规范——所有配图的配色、字体、坐标轴与图例口径的唯一权威。生成或修改任何配图、觉得图丑、要新增一类图、换配色、或篇目 README 的截图节需要回填时使用。触发：「出图」「画图」「配图」「这张图」「图好丑」「改配色」「新增一种图」「plot.py」「plot_style」。
---

# 统一绘图规范

本系列所有图共用一套 Nature/Science 期刊风格。**图和 CSV 一样是数据产物，由脚本生成，不手画、不手工调色。** 违反这条，图就会和文章引用的 commit 对不上。

## 唯一入口

| 文件 | 角色 |
|------|------|
| `tools/plot_style.py` | **配色与样式的唯一真源**：Okabe-Ito 色盲安全调色板、斜纹冗余编码、字体与 rcParams。禁止在别处重声明颜色、字体、刻度、hatch |
| `tools/plot.py` | 标准渲染器：读 `results/{NN}-{slug}/*.csv`，产出四张图（cold/hot × latency/bandwidth）；参数扫描类数据改出 sweep 线图（cold/hot 各一张，每机器一个子图） |

```bash
python3 tools/plot.py results/{NN}-{slug}/*.csv -o figures/{NN}-{slug}/
```

- 带宽受限篇目看 `-bandwidth.png`（带理论峰值与 streaming 天花板参考线），算力受限看 `-latency.png`
- 一个变体有多个 shape（扫描）时用 `--view sweep`（脚本检测到会自动切换）：`--metric gbps` 带参考线，`--metric speedup --baseline <variant>` 画相对比值（未锁频时用这个）
- 改动 `tools/plot_style.py` 后**必须重跑所有已存在的图**，否则系列内风格不一致

## 风格是什么（改样式前先读）

- 白底、无网格、无 chartjunk；完整边框 + 四边内向主/次刻度，0.8 pt 细线
- sans-serif（Nature 要求），优先 Helvetica/Arial，回退 DejaVu Sans
- **Okabe-Ito 色盲安全调色板 + 每系列不同斜纹**：绝不只靠颜色区分系列（Color Universal Design），灰度打印与色觉障碍读者都能分辨
- 数值直接标在数据上；图例放到坐标区外（右侧），不得压住柱、数值标签或参考线
- 系列顺序固定为 `read / copy / write / baseline / optimized`（`VARIANT_ORDER`），CSV 增删行不会让图重排

## 新增一类图 / 新增一个篇目图

1. **不新建调色逻辑**：`from plot_style import apply_style, variant_style`，绘制前 `apply_style()`
2. 画柱时用 `variant_style(variants)` 返回的 `color` 与 `hatch` 成对赋值；画线时用 `color` + `marker` + `linestyle`，同样不只靠颜色区分
3. 参考线、误差棒用 `INK`；误差棒口径固定 p10–p90，图上注明
4. 新篇目出图后，把图嵌进 `kernels/{NN}-{slug}/README.md` 的「截图」节（相对路径 `../../figures/{NN}-{slug}/...`）

## 硬规则

- 图只能由 CSV 生成；**没有 CSV 支撑的图不许进文章**
- cold / hot 分开画，不混一张图（见 `docs/measurement-methodology.md`）
- 不为了让图好看而改数据、裁分位数、丢掉负结果行
- 改样式或新增图类型后，同步 `tools/plot.py` 顶部说明与本 skill，别让文档和代码分叉

## 完成判据

- [ ] 图由 `tools/plot.py`（或调用 `plot_style` 的脚本）生成，无手改像素
- [ ] 配色/斜纹来自 `plot_style`，无本地硬编码颜色
- [ ] 图例未遮挡数据，数值标签可读
- [ ] 篇目 README 截图节已更新，且与这次生成的 PNG 一致
