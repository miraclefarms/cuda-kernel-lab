#pragma once

// Measurement-side CSV emission.
//
// 测量侧 CSV 输出。二进制只负责写它确知的东西：跑了什么、耗时多久。
// 环境列（机器代号、驱动、toolkit、实际时钟、MIG、是否独占、git commit）由
// tools/run.py 读取 nvidia-smi/git 后前置拼上。这样拆分的意义是：kernel 二进制
// 永远不会报告自己没验证过的环境。
//
// The binary emits only what it can know for certain: what it ran and how long
// it took. Environment columns (machine, driver, toolkit, actual clocks, MIG,
// exclusivity, git commit) are prepended by tools/run.py, which can read nvidia-smi
// and git. Keeping the split here means a kernel binary never reports an
// environment it did not actually verify.
//
// Schema is fixed. tools/plot.py and docs/measurement-methodology.md depend on it.
// 列定义固定，tools/plot.py 与 docs/measurement-methodology.md 依赖它。

#include <cstdio>
#include <string>

#include "bench/timing.cuh"

namespace bench {

// 一行 CSV 记录：描述一次「变体 × 口径 × 形状」的测量。
struct RunRow {
  std::string kernel;   // 篇号 slug，如 "00-template"
  std::string variant;  // "baseline" | "optimized" | 自由命名
  std::string mode;     // "cold" | "hot"
  std::string shape;    // 问题规模，如 "n=67108864"
  std::string dtype;    // 数据类型，如 "fp32"
  double bytes = 0.0;   // 单次调用搬动的字节数，不适用时为 0
  double flops = 0.0;   // 单次调用完成的浮点运算数，不适用时为 0
  Stats stats;
};

inline void print_csv_header() {
  std::printf(
      "kernel,variant,mode,shape,dtype,bytes,flops,samples,min_ms,p10_ms,median_ms,p90_ms,max_ms\n");
}

inline void print_csv_row(const RunRow& r) {
  std::printf("%s,%s,%s,%s,%s,%.0f,%.0f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", r.kernel.c_str(),
              r.variant.c_str(), r.mode.c_str(), r.shape.c_str(), r.dtype.c_str(), r.bytes,
              r.flops, r.stats.samples, r.stats.min_ms, r.stats.p10_ms, r.stats.median_ms,
              r.stats.p90_ms, r.stats.max_ms);
}

}  // namespace bench
