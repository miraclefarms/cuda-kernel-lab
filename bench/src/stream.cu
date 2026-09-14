#include "bench/stream.cuh"

namespace bench {
namespace {

// Four independent accumulator lanes rather than one: a single chain of
// dependent FP adds can throttle a read-only kernel below the memory system,
// which would understate the very ceiling this exists to establish.
__global__ void bw_read(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = src[i];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  // Unreachable for real data; exists only so the loads cannot be eliminated.
  if (acc.x == -1.0f && acc.y == -1.0f) sink[blockIdx.x] = acc.z + acc.w;
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

}  // namespace

StreamCeiling measure_stream_ceiling(int device, int warmup, int samples, int batch) {
  cudaDeviceProp prop{};
  BENCH_CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t bytes = kStreamBufferBytes;
  const size_t n4 = bytes / sizeof(float4);

  float4* a = nullptr;
  float4* b = nullptr;
  float* sink = nullptr;
  BENCH_CHECK(cudaMalloc(&a, bytes));
  BENCH_CHECK(cudaMalloc(&b, bytes));
  BENCH_CHECK(cudaMalloc(&sink, sizeof(float) * 65536));
  BENCH_CHECK(cudaMemset(a, 1, bytes));

  cudaStream_t stream{};
  BENCH_CHECK(cudaStreamCreate(&stream));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;

  auto run_read = [&](cudaStream_t st) { bw_read<<<grid, block, 0, st>>>(a, sink, n4); };
  auto run_copy = [&](cudaStream_t st) { bw_copy<<<grid, block, 0, st>>>(b, a, n4); };
  auto run_write = [&](cudaStream_t st) { bw_write<<<grid, block, 0, st>>>(a, n4); };

  StreamCeiling out;
  out.read.bytes = static_cast<double>(bytes);
  out.copy.bytes = static_cast<double>(bytes) * 2.0;
  out.write.bytes = static_cast<double>(bytes);

  out.read.cold = measure_cold(run_read, warmup, samples, stream, device);
  out.read.hot = measure_hot(run_read, warmup, samples, batch, stream);
  out.copy.cold = measure_cold(run_copy, warmup, samples, stream, device);
  out.copy.hot = measure_hot(run_copy, warmup, samples, batch, stream);
  out.write.cold = measure_cold(run_write, warmup, samples, stream, device);
  out.write.hot = measure_hot(run_write, warmup, samples, batch, stream);

  BENCH_CHECK(cudaStreamDestroy(stream));
  BENCH_CHECK(cudaFree(a));
  BENCH_CHECK(cudaFree(b));
  BENCH_CHECK(cudaFree(sink));
  return out;
}

}  // namespace bench
