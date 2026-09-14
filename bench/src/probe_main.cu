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
double theoretical_gbps(const cudaDeviceProp& p) {
  // memoryClockRate is in kHz, memoryBusWidth in bits, DDR -> x2.
  return 2.0 * static_cast<double>(p.memoryClockRate) * 1e3 *
         (static_cast<double>(p.memoryBusWidth) / 8.0) / 1e9;
}

void print_kv(const char* k, const std::string& v) { std::printf("%-28s %s\n", k, v.c_str()); }
void print_kv(const char* k, long long v) { std::printf("%-28s %lld\n", k, v); }
void print_kv(const char* k, double v) { std::printf("%-28s %.2f\n", k, v); }

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
  print_kv("theoretical BW (GB/s)", theoretical_gbps(p));
  print_kv("clock rate (MHz)", static_cast<double>(p.clockRate) / 1e3);
  print_kv("memory clock (MHz)", static_cast<double>(p.memoryClockRate) / 1e3);
  print_kv("cooperative launch", static_cast<long long>(p.cooperativeLaunch));
  print_kv("async engine count", static_cast<long long>(p.asyncEngineCount));
  print_kv("unified addressing", static_cast<long long>(p.unifiedAddressing));
  print_kv("ECC enabled", static_cast<long long>(p.ECCEnabled));
  print_kv("compute mode", static_cast<long long>(p.computeMode));

  // Feature gates this series actually cares about. cc >= 9.0 is the line where
  // TMA, thread block clusters, DSMEM and wgmma appear.
  const bool hopper_plus = (p.major >= 9);
  const bool blackwell_plus = (p.major >= 10);
  print_kv("TMA / cluster / wgmma", std::string(hopper_plus ? "yes (cc >= 9.0)" : "no"));
  print_kv("tcgen05 / TMEM", std::string(blackwell_plus ? "yes (cc >= 10.0)" : "no"));

  std::printf(
      "\nnote: peak FLOP/s is not queryable from the runtime; take it from the vendor spec\n"
      "      and record it in docs/environment-matrix.md rather than guessing here.\n");
  return 0;
}
