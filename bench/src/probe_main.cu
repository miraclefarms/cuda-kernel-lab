// bench-probe — baseline portrait of whatever GPU this is running on.
//
// This is the tool behind post 01: the three machines in this project differ far
// more in compute:bandwidth ratio than in ISA, and that ratio is what decides
// whether an optimization pays off. Print it before arguing about anything else.

#include <cstdio>
#include <string>

#include "bench/timing.cuh"

namespace {

// Peak FLOP/s is not queryable, so ridge point is computed from measured-ish
// theoretical bandwidth and a caller-supplied peak. Anything we cannot verify on
// device is left to the article, not fabricated here.
//
// CUDA 13 removed clockRate / memoryClockRate / computeMode from cudaDeviceProp;
// they are only reachable through cudaDeviceGetAttribute now.
int device_attr(cudaDeviceAttr attr, int device) {
  int value = 0;
  BENCH_CHECK(cudaDeviceGetAttribute(&value, attr, device));
  return value;
}

double theoretical_gbps(const cudaDeviceProp& p, int device) {
  // Memory clock is in kHz, bus width in bits, DDR -> x2.
  const int mem_khz = device_attr(cudaDevAttrMemoryClockRate, device);
  return 2.0 * static_cast<double>(mem_khz) * 1e3 *
         (static_cast<double>(p.memoryBusWidth) / 8.0) / 1e9;
}

void print_kv(const char* k, const std::string& v) { std::printf("%-28s %s\n", k, v.c_str()); }
void print_kv(const char* k, long long v) { std::printf("%-28s %lld\n", k, v); }
void print_kv(const char* k, double v) { std::printf("%-28s %.2f\n", k, v); }

// Measured streaming bandwidth. The working set is far past this class of L2
// (tens of MiB), so what these kernels report is HBM, not cache. float4 is the
// widest access this pattern gets without TMA; the vendor's theoretical number
// says nothing about how close a real kernel gets to it, which is the point.
constexpr size_t kFloats = 64u << 20;  // 64 Mi floats = 256 MiB per buffer

__global__ void bw_read(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
  float acc = 0.0f;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == -1.0f) sink[blockIdx.x] = acc;  // unreachable; keeps the loads from being dropped
}

__global__ void bw_copy(float4* __restrict__ dst, const float4* __restrict__ src, size_t n4) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void bw_write(float4* __restrict__ dst, size_t n4) {
  const float4 v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    dst[i] = v;
  }
}

template <typename Launch>
double stream_gbps(Launch&& launch, double bytes_per_iter, int iters, cudaStream_t stream) {
  for (int i = 0; i < 3; ++i) launch(stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{}, stop{};
  BENCH_CHECK(cudaEventCreate(&start));
  BENCH_CHECK(cudaEventCreate(&stop));
  BENCH_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) launch(stream);
  BENCH_CHECK(cudaEventRecord(stop, stream));
  BENCH_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  BENCH_CHECK(cudaEventElapsedTime(&ms, start, stop));
  BENCH_CHECK(cudaEventDestroy(start));
  BENCH_CHECK(cudaEventDestroy(stop));
  return bytes_per_iter * iters / (static_cast<double>(ms) / 1e3) / 1e9;
}

void print_bandwidth(const cudaDeviceProp& p, double theoretical) {
  const size_t n4 = kFloats / 4;
  const size_t bytes = kFloats * sizeof(float);
  float4* a = nullptr;
  float4* b = nullptr;
  float* sink = nullptr;
  BENCH_CHECK(cudaMalloc(&a, bytes));
  BENCH_CHECK(cudaMalloc(&b, bytes));
  BENCH_CHECK(cudaMalloc(&sink, sizeof(float) * 4096));
  BENCH_CHECK(cudaMemset(a, 1, bytes));

  cudaStream_t stream{};
  BENCH_CHECK(cudaStreamCreate(&stream));
  const int block = 256;
  const int grid = p.multiProcessorCount * 32;
  const int iters = 20;

  const double read = stream_gbps(
      [&](cudaStream_t st) { bw_read<<<grid, block, 0, st>>>(a, sink, n4); },
      static_cast<double>(bytes), iters, stream);
  const double copy = stream_gbps(
      [&](cudaStream_t st) { bw_copy<<<grid, block, 0, st>>>(b, a, n4); },
      static_cast<double>(bytes) * 2.0, iters, stream);
  const double write = stream_gbps(
      [&](cudaStream_t st) { bw_write<<<grid, block, 0, st>>>(a, n4); },
      static_cast<double>(bytes), iters, stream);

  const auto pct = [theoretical](double v) { return 100.0 * v / theoretical; };
  std::printf("\n-- measured streaming bandwidth (%zu MiB working set, %d iters) --\n",
              bytes >> 20, iters);
  std::printf("%-12s %8.1f GB/s   %5.1f%% of theoretical\n", "read", read, pct(read));
  std::printf("%-12s %8.1f GB/s   %5.1f%% of theoretical\n", "copy", copy, pct(copy));
  std::printf("%-12s %8.1f GB/s   %5.1f%% of theoretical\n", "write", write, pct(write));

  BENCH_CHECK(cudaStreamDestroy(stream));
  BENCH_CHECK(cudaFree(a));
  BENCH_CHECK(cudaFree(b));
  BENCH_CHECK(cudaFree(sink));
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  if (argc > 1) device = std::atoi(argv[1]);
  BENCH_CHECK(cudaSetDevice(device));

  cudaDeviceProp p{};
  BENCH_CHECK(cudaGetDeviceProperties(&p, device));

  int driver = 0;
  int runtime = 0;
  BENCH_CHECK(cudaDriverGetVersion(&driver));
  BENCH_CHECK(cudaRuntimeGetVersion(&runtime));
  const double theoretical = theoretical_gbps(p, device);

  print_kv("device", std::string(p.name));
  print_kv("compute capability", std::to_string(p.major) + "." + std::to_string(p.minor));
  print_kv("driver version", static_cast<long long>(driver));
  print_kv("runtime version", static_cast<long long>(runtime));
  print_kv("SM count", static_cast<long long>(p.multiProcessorCount));
  print_kv("max threads / SM", static_cast<long long>(p.maxThreadsPerMultiProcessor));
  print_kv("regs / SM", static_cast<long long>(p.regsPerMultiprocessor));
  print_kv("smem / SM (KiB)", static_cast<long long>(p.sharedMemPerMultiprocessor / 1024));
  print_kv("smem / block opt-in (KiB)",
           static_cast<long long>(p.sharedMemPerBlockOptin / 1024));
  print_kv("L2 (MiB)", static_cast<long long>(p.l2CacheSize / (1024 * 1024)));
  print_kv("global memory (GiB)",
           static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  print_kv("memory bus (bit)", static_cast<long long>(p.memoryBusWidth));
  print_kv("theoretical BW (GB/s)", theoretical);
  print_kv("clock rate (MHz)", static_cast<double>(device_attr(cudaDevAttrClockRate, device)) / 1e3);
  print_kv("memory clock (MHz)",
           static_cast<double>(device_attr(cudaDevAttrMemoryClockRate, device)) / 1e3);
  print_kv("cooperative launch", static_cast<long long>(p.cooperativeLaunch));
  print_kv("async engine count", static_cast<long long>(p.asyncEngineCount));
  print_kv("unified addressing", static_cast<long long>(p.unifiedAddressing));
  print_kv("ECC enabled", static_cast<long long>(p.ECCEnabled));
  print_kv("compute mode", static_cast<long long>(device_attr(cudaDevAttrComputeMode, device)));

  // Feature gates this series actually cares about. cc >= 9.0 is the line where
  // TMA, thread block clusters, DSMEM and wgmma appear.
  const bool hopper_plus = (p.major >= 9);
  const bool blackwell_plus = (p.major >= 10);
  print_kv("TMA / cluster / wgmma", std::string(hopper_plus ? "yes (cc >= 9.0)" : "no"));
  print_kv("tcgen05 / TMEM", std::string(blackwell_plus ? "yes (cc >= 10.0)" : "no"));

  print_bandwidth(p, theoretical);

  std::printf(
      "\nnote: peak FLOP/s is not queryable from the runtime; take it from the vendor spec\n"
      "      and record it in docs/environment-matrix.md rather than guessing here.\n");
  return 0;
}
