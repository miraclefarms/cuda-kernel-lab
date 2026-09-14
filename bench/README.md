# bench/

全系列共用的测量基础设施。改这里等于改所有已发表数字的口径，改动必须同步更新
`docs/measurement-methodology.md`，并在文章里说明。

- `include/bench/timing.cuh` — `measure_cold` / `measure_hot` 两种口径、L2 flush、分位数统计
- `include/bench/csv.hpp` — 测量侧 CSV schema（环境列由 `tools/run.py` 补齐）
- `src/probe_main.cu` — `bench-probe`，打印本机基线画像，并实测 streaming
  带宽（read / copy / write）与标称值对比

## 为什么分 cold / hot

`cold` 每次 flush L2 再计时一次启动，反映「一层只跑一次」的真实延迟；`hot` 把 `batch`
次启动录进 CUDA Graph 重放，摊掉启动开销与 host 抖动，反映 decode 循环里连续重放的吞吐。
两者在带宽受限 kernel 上能差几倍。混在一张图里的数字没有意义。

## 为什么必须 flush L2

`cudaDeviceSynchronize` 不会清 L2。不 flush 的话，重复读同一块输入的 kernel 测到的是 L2
带宽，报出来的 HBM 数字会高出数倍——这是 CUDA microbenchmark 最常见的一种说谎方式。
