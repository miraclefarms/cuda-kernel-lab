#pragma once

// Timing primitives shared by every kernel in this repo.
//
// Two measurement modes, always reported separately:
//   cold — L2 flushed before each sample, exactly one launch timed. This is the
//          number that matters for a kernel that runs once per layer.
//   hot  — `batch` launches replayed from a CUDA graph, launch overhead and host
//          jitter amortized. This is the number that matters inside a decode loop.
//
// Mixing the two produces a figure that means nothing, so the API keeps them apart.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bench {

inline void check(cudaError_t e, const char* expr, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[CUDA] %s failed at %s:%d: %s\n", expr, file, line,
                 cudaGetErrorString(e));
    std::exit(1);
  }
}

#define BENCH_CHECK(expr) ::bench::check((expr), #expr, __FILE__, __LINE__)

struct Stats {
  double min_ms = 0.0;
  double p10_ms = 0.0;
  double median_ms = 0.0;
  double p90_ms = 0.0;
  double max_ms = 0.0;
  int samples = 0;
};

// Quantiles, never a mean: on a shared or unlocked-clock machine the tail is
// information, and an average silently folds it into the headline number.
inline Stats summarize(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  const auto pick = [&v](double q) {
    const double idx = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
  };
  s.min_ms = v.front();
  s.p10_ms = pick(0.10);
  s.median_ms = pick(0.50);
  s.p90_ms = pick(0.90);
  s.max_ms = v.back();
  s.samples = static_cast<int>(v.size());
  return s;
}

// cudaDeviceSynchronize does not evict L2. Without this, a kernel that re-reads
// its input measures L2 bandwidth and reports an HBM number that is off by
// several times — the single most common way a CUDA microbenchmark lies.
class L2Flusher {
 public:
  explicit L2Flusher(int device = 0) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, device));
    bytes_ = static_cast<size_t>(prop.l2CacheSize) * 2;
    if (bytes_ == 0) bytes_ = static_cast<size_t>(64) << 20;
    BENCH_CHECK(cudaMalloc(&buf_, bytes_));
  }

  ~L2Flusher() {
    if (buf_ != nullptr) cudaFree(buf_);
  }

  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  void flush(cudaStream_t stream) { BENCH_CHECK(cudaMemsetAsync(buf_, 0, bytes_, stream)); }

  size_t bytes() const { return bytes_; }

 private:
  void* buf_ = nullptr;
  size_t bytes_ = 0;
};

// `body` must enqueue work on the stream it is handed and nothing else — no
// synchronization, no allocation. Everything else here depends on that.
template <typename Body>
Stats measure_cold(Body&& body, int warmup, int samples, cudaStream_t stream, int device = 0) {
  L2Flusher flusher(device);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) body(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    flusher.flush(stream);
    BENCH_CHECK(cudaEventRecord(start, stream));
    body(stream);
    BENCH_CHECK(cudaEventRecord(stop, stream));
    BENCH_CHECK(cudaEventSynchronize(stop));
    float dt = 0.0f;
    BENCH_CHECK(cudaEventElapsedTime(&dt, start, stop));
    ms.push_back(static_cast<double>(dt));
  }

  BENCH_CHECK(cudaEventDestroy(start));
  BENCH_CHECK(cudaEventDestroy(stop));
  return summarize(std::move(ms));
}

template <typename Body>
Stats measure_hot(Body&& body, int warmup, int samples, int batch, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) body(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  BENCH_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < batch; ++i) body(stream);
  BENCH_CHECK(cudaStreamEndCapture(stream, &graph));
  BENCH_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));

  BENCH_CHECK(cudaGraphLaunch(exec, stream));
  BENCH_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    BENCH_CHECK(cudaEventRecord(start, stream));
    BENCH_CHECK(cudaGraphLaunch(exec, stream));
    BENCH_CHECK(cudaEventRecord(stop, stream));
    BENCH_CHECK(cudaEventSynchronize(stop));
    float dt = 0.0f;
    BENCH_CHECK(cudaEventElapsedTime(&dt, start, stop));
    ms.push_back(static_cast<double>(dt) / static_cast<double>(batch));
  }

  BENCH_CHECK(cudaEventDestroy(start));
  BENCH_CHECK(cudaEventDestroy(stop));
  BENCH_CHECK(cudaGraphExecDestroy(exec));
  BENCH_CHECK(cudaGraphDestroy(graph));
  return summarize(std::move(ms));
}

}  // namespace bench
