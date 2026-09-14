#pragma once

// Measurement-side CSV emission.
//
// The binary emits only what it can know for certain: what it ran and how long
// it took. Environment columns (machine, driver, toolkit, actual clocks, MIG,
// exclusivity, git commit) are prepended by tools/run.py, which can read nvidia-smi
// and git. Keeping the split here means a kernel binary never reports an
// environment it did not actually verify.
//
// Schema is fixed. tools/plot.py and docs/measurement-methodology.md depend on it.

#include <cstdio>
#include <string>

#include "bench/timing.cuh"

namespace bench {

struct RunRow {
  std::string kernel;   // e.g. "00-template"
  std::string variant;  // "baseline" | "optimized" | free-form
  std::string mode;     // "cold" | "hot"
  std::string shape;    // e.g. "n=67108864"
  std::string dtype;    // e.g. "fp32"
  double bytes = 0.0;   // bytes moved by one invocation, 0 if not applicable
  double flops = 0.0;   // flops performed by one invocation, 0 if not applicable
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
