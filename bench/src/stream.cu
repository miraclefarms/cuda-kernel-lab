#include "bench/stream.cuh"

namespace bench {
namespace {

// Four independent accumulator lanes rather than one: a single chain of
// dependent FP adds can throttle a read-only kernel below the memory system,
// which would understate the very ceiling this exists to establish.
//
// 纯读带宽。用 4 条互相独立的累加链（x/y/z/w），而不是一条：单链浮点加法有
// 依赖延迟，会把只读 kernel 拖到内存系统之下，从而低估本要测的上限。
// 每个线程以 float4（128-bit）加 grid-stride 循环读取。
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
  // 对真实数据永远不成立，仅用于防止编译器把整段 load 优化掉。
  if (acc.x == -1.0f && acc.y == -1.0f) sink[blockIdx.x] = acc.z + acc.w;
}

// 拷贝带宽：读 src、写 dst，一次搬 2x 流量，用来测读写混合下的上限。
__global__ void bw_copy(float4* __restrict__ dst, const float4* __restrict__ src, size_t n4) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    dst[i] = src[i];
  }
}

// 纯写带宽：只写不读，测写方向的上限。
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
  const size_t n4 = bytes / sizeof(float4);  // 元素数按 float4 计

  // a 既是读的源、也是写/拷贝的目标；b 仅作拷出目标。
  // sink 很小，只为了让 bw_read 的累加结果有个出口，防止被优化掉。
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
  // grid 按设备规模给（每 SM 32 个 block），与问题大小无关；
  // 带宽瓶颈在内核里靠 grid-stride 循环覆盖。
  const int grid = prop.multiProcessorCount * 32;

  auto run_read = [&](cudaStream_t st) { bw_read<<<grid, block, 0, st>>>(a, sink, n4); };
  auto run_copy = [&](cudaStream_t st) { bw_copy<<<grid, block, 0, st>>>(b, a, n4); };
  auto run_write = [&](cudaStream_t st) { bw_write<<<grid, block, 0, st>>>(a, n4); };

  StreamCeiling out;
  // 各模式的流量：copy 读写各一遍，所以是 1x 的 2 倍。
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
