# 复现

## 前置

- CUDA Toolkit 13.x（`nvcc --version` 确认）
- CMake ≥ 3.24
- Python 3.10+，出图需要 `matplotlib`
- Ampere 或更新的 GPU

## 构建

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a   # H20 / H200
# cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80  # A100
cmake --build build -j
```

架构必须显式指定。默认值会让「这个数字是哪个架构跑出来的」变得不可追溯，所以
CMakeLists 里直接报错而不是给默认值。`90a` 的 `a` 后缀不能省——`wgmma`、TMA、cluster
都是 arch-conditional 特性，缺后缀会静默不可用。

## 跑一次

```bash
./build/bench/bench-probe                                  # 先看本机画像
python3 tools/run.py --kernel 00-template --machine h200   # 写 results/00-template/
python3 tools/plot.py results/00-template/*.csv -o figures/00-template/
```

拿得到 root 时先锁频，并把这一事实告诉 run.py：

```bash
sudo nvidia-smi -lgc <min>,<max>
python3 tools/run.py --kernel 00-template --machine h200 --clocks-locked
sudo nvidia-smi -rgc
```

## 常见坑

- **数字比预期高几倍** → 多半没 flush L2，或者 shape 小到装得进 L2。把 `--n` 调大。
- **cold 与 hot 差很多** → 正常。小 kernel 上 cold 主要是启动开销与冷缓存。
- **同一命令两次跑差 20%+** → 机器没独占或没锁频。看 CSV 里的 `sm_clock_mhz` 与
  `clocks_locked`，结论改用相对比值。
- **ncu 报权限错误** → 需要 root 或 `NVreg_RestrictProfilingToAdminUsers=0`。
  拿不到就在文章里声明本篇无计数器证据。
- **`git_dirty=yes`** → 这份数据不能进文章。提交干净后重跑。

## 跑出了不同结论

开 issue，附上 CSV（它自带环境列）与 `bench-probe` 输出。不同机器上结论不同正是这个
仓库想收集的东西。
