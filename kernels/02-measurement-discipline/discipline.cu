// 02-measurement-discipline — one kernel, four ways to time it.
//
// 02-measurement-discipline：同一个 kernel，四种计时方法。
// kernel 本身刻意不变（00-template 的 float4 triad），变的只有「怎么量」。这篇要证明的是：
// 大量不可复现的 CUDA 性能结论，问题不在 kernel，而在测量——没 flush L2、把批量重放
// 当单次延迟、没有预热、只报均值。所以这里的「baseline」是错误的测法，「optimized」是
// bench/ 里的正确测法，两者测的是同一个 kernel、同一份输入。
//
// The kernel is deliberately held fixed (the float4 triad from 00-template); only
// the timing protocol changes. The undisciplined protocols below are wrong on
// purpose and are labelled as such in the CSV — they are the experiment, not a
// convenience. Everything that is not a demonstration of a mistake goes through
// bench::measure_cold / measure_hot.
//
// 扫描维度是单个缓冲区大小，从 L2 的几分之一扫到 bench::kStreamBufferBytes：工作集
// 装得进 L2 时，不 flush 的测法会报出高于 HBM 标称带宽的「虚假带宽」——第 1 节的数据
// 就是这条曲线越过理论峰值的那一段。

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "bench/csv.hpp"
#include "bench/stream.cuh"
#include "bench/timing.cuh"

namespace {

// 与 00-template 的 optimized 完全相同：float4 访存、grid 按 SM 数配。
// Identical to 00-template's optimized variant.
__global__ void triad_vectorized(float4* __restrict__ a, const float4* __restrict__ b,
                                 const float4* __restrict__ c, float s, size_t n4) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n4; i += stride) {
    const float4 bv = b[i];
    const float4 cv = c[i];
    float4 av;
    av.x = bv.x + s * cv.x;
    av.y = bv.y + s * cv.y;
    av.z = bv.z + s * cv.z;
    av.w = bv.w + s * cv.w;
    a[i] = av;
  }
}

// ---------------------------------------------------------------------------
// 错误测法。每一个都对应一种在公开 benchmark 里真实出现过的写法，不要拿去测别的东西。
// Wrong on purpose. Each mirrors a harness that shows up in published numbers.
// ---------------------------------------------------------------------------

// no-flush：计时方式与 bench::measure_cold 完全一致（事件对、预热、单次启动），唯一的
// 差别是采样之间不 flush L2。工作集装得进 L2 时，测到的是 L2 带宽。
// no-flush: identical to measure_cold except the L2 is never flushed between samples.
template <typename Body>
bench::Stats time_no_flush(Body&& body, int warmup, int samples, cudaStream_t stream,
                           std::vector<double>* raw) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) body(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
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
  *raw = ms;
  return bench::summarize(std::move(ms));
}

// batched：一个事件对里连续启动 `batch` 次、不 flush，再除以 batch，当成「单次冷启动
// 延迟」报出去。这是冷热口径混用——本仓库的 bench-probe 早期版本就是这么测的，报出的
// 「天花板」被 00-template 超了 6%。
// batched: `batch` back-to-back launches inside one event pair, no flush, divided
// by batch and reported as single-launch latency. This is how the first version of
// bench-probe measured its "ceiling".
template <typename Body>
bench::Stats time_batched(Body&& body, int warmup, int samples, int batch, cudaStream_t stream,
                          std::vector<double>* raw) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) body(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    BENCH_CHECK(cudaEventRecord(start, stream));
    for (int k = 0; k < batch; ++k) body(stream);
    BENCH_CHECK(cudaEventRecord(stop, stream));
    BENCH_CHECK(cudaEventSynchronize(stop));
    float dt = 0.0f;
    BENCH_CHECK(cudaEventElapsedTime(&dt, start, stop));
    ms.push_back(static_cast<double>(dt) / static_cast<double>(batch));
  }
  BENCH_CHECK(cudaEventDestroy(start));
  BENCH_CHECK(cudaEventDestroy(stop));
  *raw = ms;
  return bench::summarize(std::move(ms));
}

// wallclock：host 侧 std::chrono 包住「启动 + 同步」，不预热、不 flush。它把 host 侧启动、
// 同步等待与 OS 调度抖动全算进 kernel 耗时；这些开销是长尾，均值会被拖走，分位数不会——
// 第 4 节用 --samples-out 的原始样本展示这一点。
// wallclock: host clock around launch + synchronize, no warmup, no flush. Host launch
// cost, the synchronize wait and OS jitter all land in the number as a long tail.
template <typename Body>
bench::Stats time_wallclock(Body&& body, int samples, cudaStream_t stream,
                            std::vector<double>* raw) {
  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    body(stream);
    BENCH_CHECK(cudaStreamSynchronize(stream));
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  *raw = ms;
  return bench::summarize(std::move(ms));
}

// 前 n 个元素与 host 黄金结果逐一比对。输入是小整数，结果在 float 里精确可表示。
bool verify(const std::vector<float>& got, const std::vector<float>& b, const std::vector<float>& c,
            float s, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const float want = b[i] + s * c[i];
    if (std::abs(got[i] - want) > 1e-4f * std::max(1.0f, std::abs(want))) {
      std::fprintf(stderr, "[verify] mismatch at %zu: got %f want %f\n", i, got[i], want);
      return false;
    }
  }
  return true;
}

std::vector<size_t> parse_sizes_kib(const std::string& list) {
  std::vector<size_t> out;
  size_t pos = 0;
  while (pos < list.size()) {
    const size_t end = std::min(list.find(',', pos), list.size());
    out.push_back(std::strtoull(list.substr(pos, end - pos).c_str(), nullptr, 10) << 10);
    pos = end + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  int warmup = 5;
  int samples = 50;
  int batch = 20;
  std::string sizes_arg;
  std::string samples_out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
    else if (arg == "--batch" && i + 1 < argc) batch = std::atoi(argv[++i]);
    else if (arg == "--sizes-kib" && i + 1 < argc) sizes_arg = argv[++i];
    else if (arg == "--samples-out" && i + 1 < argc) samples_out = argv[++i];
  }

  cudaDeviceProp prop{};
  BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
  const size_t l2 = prop.l2CacheSize;

  // 缓冲区大小从 L2 推导，不写常数：三台机器 L2 是 40 / 60 / 待测 MiB，写死的尺寸在一台上
  // 装得进 L2、在另一台上装不进，曲线的拐点就不可比。最后一档是全仓库统一的工作集。
  // Sizes derive from this device's L2, then end at the shared working set.
  std::vector<size_t> sizes;
  if (!sizes_arg.empty()) {
    sizes = parse_sizes_kib(sizes_arg);
  } else {
    for (const double f : {1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0, 2.0}) {
      sizes.push_back(static_cast<size_t>(static_cast<double>(l2) * f));
    }
    sizes.push_back(bench::kStreamBufferBytes);
  }
  // float4 访存要求字节数是 16 的倍数。
  for (auto& sz : sizes) sz = std::max<size_t>(16, (sz / 16) * 16);
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  const size_t max_bytes = sizes.back();
  const size_t max_n = max_bytes / sizeof(float);

  const float s = 3.0f;
  std::vector<float> hb(max_n), hc(max_n), ha(max_n);
  for (size_t i = 0; i < max_n; ++i) {
    hb[i] = static_cast<float>(i % 13);
    hc[i] = static_cast<float>(i % 7);
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  BENCH_CHECK(cudaMalloc(&da, max_bytes));
  BENCH_CHECK(cudaMalloc(&db, max_bytes));
  BENCH_CHECK(cudaMalloc(&dc, max_bytes));
  BENCH_CHECK(cudaMemcpy(db, hb.data(), max_bytes, cudaMemcpyHostToDevice));
  BENCH_CHECK(cudaMemcpy(dc, hc.data(), max_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream{};
  BENCH_CHECK(cudaStreamCreate(&stream));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;

  FILE* raw_fh = nullptr;
  if (!samples_out.empty()) {
    raw_fh = std::fopen(samples_out.c_str(), "w");
    if (raw_fh == nullptr) {
      std::fprintf(stderr, "cannot open %s\n", samples_out.c_str());
      return 1;
    }
    std::fprintf(raw_fh, "kernel,variant,mode,shape,sample,ms\n");
  }

  bench::print_csv_header();
  for (const size_t bytes : sizes) {
    const size_t n = bytes / sizeof(float);
    const size_t n4 = n / 4;
    auto body = [&](cudaStream_t st) {
      triad_vectorized<<<grid, block, 0, st>>>(reinterpret_cast<float4*>(da),
                                                reinterpret_cast<const float4*>(db),
                                                reinterpret_cast<const float4*>(dc), s, n4);
    };

    // Correctness before timing, at every size.
    // 每个尺寸都先校验再计时。
    BENCH_CHECK(cudaMemset(da, 0, bytes));
    body(stream);
    BENCH_CHECK(cudaStreamSynchronize(stream));
    BENCH_CHECK(cudaMemcpy(ha.data(), da, bytes, cudaMemcpyDeviceToHost));
    if (!verify(ha, hb, hc, s, n)) return 1;

    // shape 里带上「工作集 / L2」：三块缓冲区（a、b、c）加起来和 L2 比，决定装不装得进。
    char shape[96];
    std::snprintf(shape, sizeof(shape), "buf_kib=%zu;ws_over_l2=%.3f", bytes >> 10,
                  3.0 * static_cast<double>(bytes) / static_cast<double>(l2));

    bench::RunRow row;
    row.kernel = "02-measurement-discipline";
    row.shape = shape;
    row.dtype = "fp32";
    row.bytes = static_cast<double>(bytes) * 3.0;  // 两读一写
    row.flops = static_cast<double>(n) * 2.0;

    const auto emit = [&](const char* variant, const char* mode, const bench::Stats& st,
                          const std::vector<double>& raw) {
      row.variant = variant;
      row.mode = mode;
      row.stats = st;
      bench::print_csv_row(row);
      if (raw_fh != nullptr) {
        for (size_t k = 0; k < raw.size(); ++k) {
          std::fprintf(raw_fh, "%s,%s,%s,%s,%zu,%.6f\n", row.kernel.c_str(), variant, mode, shape,
                       k, raw[k]);
        }
      }
    };

    std::vector<double> raw;
    // 顺序固定：wallclock 紧跟校验那次启动之后、没有任何预热，其余测法各自预热。
    // Fixed order: wallclock runs right after the verification launch with no warmup.
    emit("wallclock", "cold", time_wallclock(body, samples, stream, &raw), raw);
    emit("no-flush", "cold", time_no_flush(body, warmup, samples, stream, &raw), raw);
    emit("batched", "cold", time_batched(body, warmup, samples, batch, stream, &raw), raw);
    emit("disciplined", "cold", bench::measure_cold(body, warmup, samples, stream, 0, &raw), raw);
    emit("disciplined", "hot", bench::measure_hot(body, warmup, samples, batch, stream, &raw), raw);
  }

  if (raw_fh != nullptr) std::fclose(raw_fh);
  BENCH_CHECK(cudaStreamDestroy(stream));
  BENCH_CHECK(cudaFree(da));
  BENCH_CHECK(cudaFree(db));
  BENCH_CHECK(cudaFree(dc));
  return 0;
}
