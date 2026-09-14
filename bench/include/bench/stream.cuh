#pragma once

// Streaming bandwidth ceiling — the denominator for every bandwidth claim.
//
// 流式带宽上限——所有带宽结论的分母。它刻意走和仓库里每个 kernel 相同的
// measure_cold / measure_hot 路径：如果上限的测法和被比较的 kernel 不一样，
// 这个「上限」就不是上限，只是一个无关数字。早期版本在一个事件对里连测 20 次
// 且不刷 L2，报出的「上限」反而被 00-template 超了 6%，就是典型的测法不一致。
// 必须同口径比较：冷对冷，热对热。
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
//
// 单个缓冲区大小固定为 256 MiB：远大于这一档 GPU 的 L2（H20 为 60 MiB），
// 保证访存落到 HBM；同时最大的卡也放得下三块。全仓库统一用它，上限与
// kernel 才可比。
constexpr size_t kStreamBufferBytes = static_cast<size_t>(256) << 20;

// 一种访问模式（读/拷/写）的冷热两组结果及对应带宽。
struct StreamResult {
  Stats cold;
  Stats hot;
  double bytes = 0.0;  // 单次调用的 HBM 流量字节数

  // 带宽 = 字节数 / 中位耗时，换算成 GB/s；耗时为 0 时返回 0 避免除零。
  double cold_gbps() const {
    return cold.median_ms > 0.0 ? bytes / (cold.median_ms / 1e3) / 1e9 : 0.0;
  }
  double hot_gbps() const {
    return hot.median_ms > 0.0 ? bytes / (hot.median_ms / 1e3) / 1e9 : 0.0;
  }
};

// 三种访问模式的上限：read 流量 1x、copy 流量 2x（读+写）、write 流量 1x。
struct StreamCeiling {
  StreamResult read;
  StreamResult copy;
  StreamResult write;

  // The ceiling to quote is the best of the three: it is the closest a trivial
  // access pattern gets to the hardware, whatever mix produced it.
  // 对外引用的上限取三者最好值：无论哪种组合，它都是最朴素的访存模式
  // 能逼近硬件的程度。
  double best_cold_gbps() const {
    return std::max(std::max(read.cold_gbps(), copy.cold_gbps()), write.cold_gbps());
  }
  double best_hot_gbps() const {
    return std::max(std::max(read.hot_gbps(), copy.hot_gbps()), write.hot_gbps());
  }
};

// 在指定设备上跑满 read/copy/write 三种模式，返回冷热两组带宽上限。
StreamCeiling measure_stream_ceiling(int device, int warmup, int samples, int batch);

}  // namespace bench
