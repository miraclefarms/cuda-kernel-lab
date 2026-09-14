// 00-template — the shape every kernel directory in this repo follows.
//
// 00-template：本仓库每个篇目目录都要照这个形状来。
// 问题本身故意选得很简单（STREAM triad，a[i] = b[i] + s * c[i]），重点不在
// kernel，而在于演示「一篇的代码必须满足的契约」：baseline、optimized 两个变体，
// 相对 baseline 的正确性校验，以及在 cold/hot 两种模式下从 stdout 输出 CSV。
//
// 复制本目录，改名成 {NN}-{slug}，再替换其中的 kernel 即可。
//
// The problem is deliberately trivial (a STREAM triad, a[i] = b[i] + s * c[i])
// because the point is not the kernel: it is to show the contract a post's code
// has to satisfy — a baseline, an optimized variant, a correctness check against
// the baseline, and CSV on stdout in both cold and hot modes.
//
// Copy this directory, rename to {NN}-{slug}, replace the kernels.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/csv.hpp"
#include "bench/stream.cuh"
#include "bench/timing.cuh"

namespace {

// baseline：一个线程处理一个元素，最朴素的写法，作为性能与正确性的参照。
__global__ void triad_baseline(float* __restrict__ a, const float* __restrict__ b,
                               const float* __restrict__ c, float s, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = b[i] + s * c[i];
}

// One thread handles four contiguous elements through a 128-bit access, and the
// grid is sized to the device rather than to the problem. On a bandwidth-bound
// kernel the win comes from issuing fewer, wider memory instructions — which is
// exactly the kind of claim this repo exists to put a number on.
//
// optimized：一个线程用 128-bit（float4）访问处理 4 个连续元素，且 grid 按设备
// 规模而非问题规模来配。带宽受限时收益来自更少、更宽的访存指令——这正是本仓库
// 要用数字去验证的那类论断。
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

// 逐元素与 CPU 端黄金结果比对，带相对误差容限；用于在上性能测试前确认 kernel 正确。
bool verify(const std::vector<float>& got, const std::vector<float>& b, const std::vector<float>& c,
            float s) {
  for (size_t i = 0; i < got.size(); ++i) {
    const float want = b[i] + s * c[i];
    if (std::abs(got[i] - want) > 1e-4f * std::max(1.0f, std::abs(want))) {
      std::fprintf(stderr, "[verify] mismatch at %zu: got %f want %f\n", i, got[i], want);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // Derived from the shared buffer size rather than written out, so a kernel's
  // numbers and bench-probe's ceiling always describe the same working set.
  // 默认规模由共用缓冲区大小推导，保证 kernel 与 bench-probe 的上限描述的是
  // 同一个工作集；可通过命令行参数覆盖。
  size_t n = bench::kStreamBufferBytes / sizeof(float);
  int warmup = 5;
  int samples = 50;
  int batch = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
    else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
    else if (arg == "--batch" && i + 1 < argc) batch = std::atoi(argv[++i]);
  }
  n = (n / 4) * 4;  // vectorized variant needs a multiple of 4

  const float s = 3.0f;
  const size_t bytes = n * sizeof(float);

  // host 端输入 b、c 用固定的小周期模式填充，结果可预期、便于复现。
  std::vector<float> hb(n), hc(n), ha(n);
  for (size_t i = 0; i < n; ++i) {
    hb[i] = static_cast<float>(i % 13);
    hc[i] = static_cast<float>(i % 7);
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  BENCH_CHECK(cudaMalloc(&da, bytes));
  BENCH_CHECK(cudaMalloc(&db, bytes));
  BENCH_CHECK(cudaMalloc(&dc, bytes));
  BENCH_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
  BENCH_CHECK(cudaMemcpy(dc, hc.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream{};
  BENCH_CHECK(cudaStreamCreate(&stream));

  cudaDeviceProp prop{};
  BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int block = 256;
  const size_t grid_baseline = (n + block - 1) / block;
  const size_t n4 = n / 4;
  const int grid_vec = prop.multiProcessorCount * 32;

  auto run_baseline = [&](cudaStream_t st) {
    triad_baseline<<<static_cast<unsigned>(grid_baseline), block, 0, st>>>(da, db, dc, s, n);
  };
  auto run_vectorized = [&](cudaStream_t st) {
    triad_vectorized<<<static_cast<unsigned>(grid_vec), block, 0, st>>>(
        reinterpret_cast<float4*>(da), reinterpret_cast<const float4*>(db),
        reinterpret_cast<const float4*>(dc), s, n4);
  };

  // Correctness before performance, every time. A fast wrong kernel has produced
  // more than one memorable benchmark result.
  // 永远先验证正确性再测性能：一个跑得快但算错的 kernel 已经制造过不止一个
  // 令人印象深刻的「结果」。
  BENCH_CHECK(cudaMemsetAsync(da, 0, bytes, stream));
  run_vectorized(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));
  BENCH_CHECK(cudaMemcpy(ha.data(), da, bytes, cudaMemcpyDeviceToHost));
  if (!verify(ha, hb, hc, s)) return 1;

  const std::string shape = "n=" + std::to_string(n);
  // triad moves two reads and one write per element.
  // 每个元素读 b、c 两遍、写 a 一遍，所以流量是字节数的 3 倍。
  const double moved = static_cast<double>(bytes) * 3.0;

  bench::print_csv_header();
  // 依次测量 baseline 与 optimized，每个变体都输出 cold、hot 两行。
  for (const auto& v : {std::pair<const char*, bool>{"baseline", true},
                        std::pair<const char*, bool>{"optimized", false}}) {
    auto body = [&](cudaStream_t st) { v.second ? run_baseline(st) : run_vectorized(st); };
    bench::RunRow row;
    row.kernel = "00-template";
    row.variant = v.first;
    row.shape = shape;
    row.dtype = "fp32";
    row.bytes = moved;
    row.flops = static_cast<double>(n) * 2.0;

    row.mode = "cold";
    row.stats = bench::measure_cold(body, warmup, samples, stream);
    bench::print_csv_row(row);

    row.mode = "hot";
    row.stats = bench::measure_hot(body, warmup, samples, batch, stream);
    bench::print_csv_row(row);
  }

  BENCH_CHECK(cudaStreamDestroy(stream));
  BENCH_CHECK(cudaFree(da));
  BENCH_CHECK(cudaFree(db));
  BENCH_CHECK(cudaFree(dc));
  return 0;
}
