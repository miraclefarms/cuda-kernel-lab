// Static occupancy study for post 01. This does not time or launch a kernel.
// The register axis is the compiler's ACTUAL register count, not the requested
// accumulator count: compiler allocation and spills cannot be guessed safely.
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

#include "occupancy_probes.cuh"

namespace {

bool check(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
  return false;
}

template <int N>
bool scan(const cudaDeviceProp& prop, int device) {
  cudaFuncAttributes attr{};
  if (!check(cudaFuncGetAttributes(&attr, register_probe<N>), "cudaFuncGetAttributes")) return false;
  if (!check(cudaFuncSetAttribute(register_probe<N>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(prop.sharedMemPerBlockOptin)),
             "cudaFuncSetAttribute")) return false;

  constexpr int block_size = 256;
  const std::array<size_t, 9> smem_kib{{0, 16, 32, 48, 64, 96, 128, 160, 192}};
  for (const size_t kib : smem_kib) {
    const size_t bytes = kib * 1024;
    if (bytes > prop.sharedMemPerBlockOptin) continue;
    int blocks = 0;
    if (!check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks, register_probe<N>, block_size, bytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor")) return false;
    const int warps = blocks * (block_size / prop.warpSize);
    const double occupancy = 100.0 * blocks * block_size / prop.maxThreadsPerMultiProcessor;
    std::printf("%s,%d,%d,%d,%zu,%d,%d,%d,%d,%.2f\n", prop.name, device,
                N, attr.numRegs, kib, attr.localSizeBytes > 0 ? 1 : 0,
                static_cast<int>(attr.localSizeBytes), blocks, warps, occupancy);
  }
  return true;
}

}  // namespace

int main() {
  constexpr int device = 0;  // logical device 0, selected by CUDA_VISIBLE_DEVICES
  cudaDeviceProp prop{};
  if (!check(cudaSetDevice(device), "cudaSetDevice") ||
      !check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) return EXIT_FAILURE;
  std::puts("gpu_name,logical_device,requested_accumulators,actual_regs_per_thread,"
            "dynamic_smem_kib,has_local_memory,local_bytes_per_thread,"
            "active_blocks_per_sm,active_warps_per_sm,theoretical_occupancy_pct");
  if (!scan<8>(prop, device) || !scan<24>(prop, device) ||
      !scan<48>(prop, device) || !scan<80>(prop, device)) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
