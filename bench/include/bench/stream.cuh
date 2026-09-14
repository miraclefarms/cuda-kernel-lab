#pragma once

// Streaming bandwidth ceiling — the denominator for every bandwidth claim.
//
// This deliberately runs through the same measure_cold / measure_hot path as
// every kernel in the repo. A ceiling measured with a different discipline than
// the kernels it is compared against is not a ceiling, it is an unrelated
// number: the first version of this probe timed 20 back-to-back launches inside
// one event pair with no L2 flush, and reported a "ceiling" that 00-template
// then exceeded by 6%.
//
// Compare like with like: cold kernel numbers against the cold ceiling, hot
// against hot.

#include <cstddef>

#include "bench/timing.cuh"

namespace bench {

// One buffer. Several times any L2 in this class (H20 carries 60 MiB) so the
// traffic is HBM, while three of them still fit on the smallest card here.
// Every bandwidth number in this repo uses this size, which is what makes the
// ceiling and the kernels comparable.
constexpr size_t kStreamBufferBytes = static_cast<size_t>(256) << 20;

struct StreamResult {
  Stats cold;
  Stats hot;
  double bytes = 0.0;  // bytes of HBM traffic per invocation

  double cold_gbps() const {
    return cold.median_ms > 0.0 ? bytes / (cold.median_ms / 1e3) / 1e9 : 0.0;
  }
  double hot_gbps() const {
    return hot.median_ms > 0.0 ? bytes / (hot.median_ms / 1e3) / 1e9 : 0.0;
  }
};

struct StreamCeiling {
  StreamResult read;   // 1x traffic
  StreamResult copy;   // 2x traffic (read + write)
  StreamResult write;  // 1x traffic

  // The ceiling to quote is the best of the three: it is the closest a trivial
  // access pattern gets to the hardware, whatever mix produced it.
  double best_cold_gbps() const {
    return std::max(std::max(read.cold_gbps(), copy.cold_gbps()), write.cold_gbps());
  }
  double best_hot_gbps() const {
    return std::max(std::max(read.hot_gbps(), copy.hot_gbps()), write.hot_gbps());
  }
};

StreamCeiling measure_stream_ceiling(int device, int warmup, int samples, int batch);

}  // namespace bench
