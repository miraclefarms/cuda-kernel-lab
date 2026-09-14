#pragma once

// Timing primitives shared by every kernel in this repo.
//
// 全仓库共用的计时原语。两种测量模式必须分开报告：
//   cold（冷启动）—— 每次采样前先刷掉 L2，只计一次 launch 的耗时。
//                    适合「每层只跑一次」的 kernel。
//   hot（热重放）—— 把 `batch` 次 launch 录进 CUDA graph 后整体重放，
//                    摊薄 launch 开销与 host 抖动。适合 decode 循环内的场景。
// 两种口径混在一起报出的数字没有意义，所以 API 层面强制分开。
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

// CUDA 调用失败即打印表达式、文件行号和错误串后退出，避免错误被静默吞掉。
inline void check(cudaError_t e, const char* expr, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[CUDA] %s failed at %s:%d: %s\n", expr, file, line,
                 cudaGetErrorString(e));
    std::exit(1);
  }
}

#define BENCH_CHECK(expr) ::bench::check((expr), #expr, __FILE__, __LINE__)

// 一次测量结果的统计量：用分位数而非均值，机器的尾延迟本身就是信息。
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
// 只报分位数不报均值：共享或未锁频时均值会把长尾悄悄平均掉。
inline Stats summarize(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  // 线性插值取分位点：先把 q 映射到 [0, n-1] 的浮点下标，再在相邻样本间插值。
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
//
// L2 冲刷器：cudaDeviceSynchronize 不会逐出 L2，所以每次冷测前必须用一块
// 大于 L2 的缓冲区做 memset，把上一轮的输入踢出缓存。否则重复读同一输入的
// kernel 实际测到的是 L2 带宽，却当成 HBM 带宽报出去，差好几倍。
class L2Flusher {
 public:
  explicit L2Flusher(int device = 0) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, device));
    // 取 2 倍 L2 容量确保能把整块 L2 覆盖掉；查询不到时退化为 64 MiB。
    bytes_ = static_cast<size_t>(prop.l2CacheSize) * 2;
    if (bytes_ == 0) bytes_ = static_cast<size_t>(64) << 20;
    BENCH_CHECK(cudaMalloc(&buf_, bytes_));
  }

  ~L2Flusher() {
    if (buf_ != nullptr) cudaFree(buf_);
  }

  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  // 在给定 stream 上异步写满整块缓冲区，借此把 L2 中原有数据挤出。
  void flush(cudaStream_t stream) { BENCH_CHECK(cudaMemsetAsync(buf_, 0, bytes_, stream)); }

  size_t bytes() const { return bytes_; }

 private:
  void* buf_ = nullptr;
  size_t bytes_ = 0;
};

// `body` must enqueue work on the stream it is handed and nothing else — no
// synchronization, no allocation. Everything else here depends on that.
//
// body 只能在传入的 stream 上入队工作，不能做同步、不能分配内存；
// 否则事件计时会被算进去，测量结果失真。
template <typename Body>
Stats measure_cold(Body&& body, int warmup, int samples, cudaStream_t stream, int device = 0) {
  L2Flusher flusher(device);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));

  // 先跑 warmup 次，把首次的初始化/缺页开销排除在计时之外。
  for (int i = 0; i < warmup; ++i) body(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    flusher.flush(stream);  // 每次采样前先清空 L2，保证读到的是 HBM
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

// 热路径：把同一个 body 连续入队 batch 次录成 CUDA graph，再整图重放。
// 这样 launch 开销和 host 侧抖动被 batch 次分摊，得到的是稳态的单次耗时。
template <typename Body>
Stats measure_hot(Body&& body, int warmup, int samples, int batch, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) body(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  // 捕获 batch 次调用生成 graph，再实例化成可重复 launch 的 exec。
  BENCH_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < batch; ++i) body(stream);
  BENCH_CHECK(cudaStreamEndCapture(stream, &graph));
  BENCH_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));

  // 先空跑一次整图，等所有资源就绪后再开始计时。
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
    // 整图耗时除以 batch，折算成单次调用的耗时。
    ms.push_back(static_cast<double>(dt) / static_cast<double>(batch));
  }

  BENCH_CHECK(cudaEventDestroy(start));
  BENCH_CHECK(cudaEventDestroy(stop));
  BENCH_CHECK(cudaGraphExecDestroy(exec));
  BENCH_CHECK(cudaGraphDestroy(graph));
  return summarize(std::move(ms));
}

}  // namespace bench
