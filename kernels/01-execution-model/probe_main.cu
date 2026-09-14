// bench-probe — baseline portrait of whatever GPU this is running on.
//
// bench-probe：对当前 GPU 做一份基线画像。它是第 01 篇的工具——本项目三台机器
// 在「算力:带宽比」上的差异远大于 ISA 差异，而这个比值决定了某项优化是否划算。
// 讨论任何优化之前，先把这份画像打出来。
//
// This is the tool behind post 01: the three machines in this project differ far
// more in compute:bandwidth ratio than in ISA, and that ratio is what decides
// whether an optimization pays off. Print it before arguing about anything else.

#include <cstdio>
#include <string>

#include "bench/csv.hpp"
#include "bench/stream.cuh"
#include "bench/timing.cuh"

namespace {

// Peak FLOP/s is not queryable, so ridge point is computed from measured-ish
// theoretical bandwidth and a caller-supplied peak. Anything we cannot verify on
// device is left to the article, not fabricated here.
//
// 峰值 FLOP/s 无法从 runtime 查询，所以 ridge point 用「近似理论带宽 + 调用方
// 提供的峰值」来算。凡是设备上验证不了的数字，一律留给文章，绝不在这里编造。
//
// CUDA 13 removed clockRate / memoryClockRate / computeMode from cudaDeviceProp;
// they are only reachable through cudaDeviceGetAttribute now.
// CUDA 13 已把 clockRate / memoryClockRate / computeMode 从 cudaDeviceProp 移除，
// 只能通过 cudaDeviceGetAttribute 取得。
int device_attr(cudaDeviceAttr attr, int device) {
  int value = 0;
  BENCH_CHECK(cudaDeviceGetAttribute(&value, attr, device));
  return value;
}

// 理论带宽 = 2(DDR) × 显存时钟(kHz→Hz) × 位宽(bit→Byte) / 1e9，单位 GB/s。
double theoretical_gbps(const cudaDeviceProp& p, int device) {
  // Memory clock is in kHz, bus width in bits, DDR -> x2.
  const int mem_khz = device_attr(cudaDevAttrMemoryClockRate, device);
  return 2.0 * static_cast<double>(mem_khz) * 1e3 *
         (static_cast<double>(p.memoryBusWidth) / 8.0) / 1e9;
}

// 三个重载：按值类型分别格式化「键 值」一行，键宽 28 列对齐。
void print_kv(const char* k, const std::string& v) { std::printf("%-28s %s\n", k, v.c_str()); }
void print_kv(const char* k, long long v) { std::printf("%-28s %lld\n", k, v); }
void print_kv(const char* k, double v) { std::printf("%-28s %.2f\n", k, v); }

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  bool csv_mode = false;
  // 命令行：--csv 走标准 CSV 输出（供 tools/run.py 采集），
  // --device N 或裸数字 N 选择设备，默认 0 号卡。
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--csv") {
      csv_mode = true;
    } else if (arg == "--device" && i + 1 < argc) {
      device = std::atoi(argv[++i]);
    } else if (!arg.empty() && arg[0] != '-') {
      device = std::atoi(arg.c_str());
    }
  }
  BENCH_CHECK(cudaSetDevice(device));

  cudaDeviceProp p{};
  BENCH_CHECK(cudaGetDeviceProperties(&p, device));

  int driver = 0;
  int runtime = 0;
  BENCH_CHECK(cudaDriverGetVersion(&driver));
  BENCH_CHECK(cudaRuntimeGetVersion(&runtime));
  const double theoretical = theoretical_gbps(p, device);

  if (!csv_mode) {
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
    print_kv("clock rate (MHz)",
             static_cast<double>(device_attr(cudaDevAttrClockRate, device)) / 1e3);
    print_kv("memory clock (MHz)",
             static_cast<double>(device_attr(cudaDevAttrMemoryClockRate, device)) / 1e3);
    print_kv("cooperative launch", static_cast<long long>(p.cooperativeLaunch));
    print_kv("async engine count", static_cast<long long>(p.asyncEngineCount));
    print_kv("unified addressing", static_cast<long long>(p.unifiedAddressing));
    print_kv("ECC enabled", static_cast<long long>(p.ECCEnabled));
    print_kv("compute mode", static_cast<long long>(device_attr(cudaDevAttrComputeMode, device)));

    // Feature gates this series actually cares about. cc >= 9.0 is the line where
    // TMA, thread block clusters, DSMEM and wgmma appear.
    // 本系列关心的特性门槛：cc >= 9.0 开始有 TMA / thread block cluster /
    // DSMEM / wgmma；cc >= 10.0（Blackwell）开始有 tcgen05 / TMEM。
    const bool hopper_plus = (p.major >= 9);
    const bool blackwell_plus = (p.major >= 10);
    print_kv("TMA / cluster / wgmma", std::string(hopper_plus ? "yes (cc >= 9.0)" : "no"));
    print_kv("tcgen05 / TMEM", std::string(blackwell_plus ? "yes (cc >= 10.0)" : "no"));
  }

  // Same measurement path as every kernel in the repo, so this number is a
  // legitimate denominator rather than a differently-measured curiosity.
  // 与仓库里每个 kernel 走同一条测量路径，这个数才是合法的分母，
  // 而不是一个口径不同的「怪值」。
  const bench::StreamCeiling bw = bench::measure_stream_ceiling(device, /*warmup=*/5,
                                                                /*samples=*/30, /*batch=*/20);

  // CSV 模式：把 read/copy/write 三种模式的冷热结果各输出两行。
  if (csv_mode) {
    bench::print_csv_header();
    const std::string shape = "buf=" + std::to_string(bench::kStreamBufferBytes >> 20) + "MiB";
    const auto emit = [&](const char* variant, const bench::StreamResult& r) {
      for (const char* mode : {"cold", "hot"}) {
        bench::RunRow out;
        out.kernel = "01-execution-model";
        out.variant = variant;
        out.mode = mode;
        out.shape = shape;
        out.dtype = "fp32";
        out.bytes = r.bytes;
        out.stats = std::string(mode) == "cold" ? r.cold : r.hot;
        bench::print_csv_row(out);
      }
    };
    emit("read", bw.read);
    emit("copy", bw.copy);
    emit("write", bw.write);
    return 0;
  }

  // 人读模式：把实测带宽换算成占理论带宽的百分比，并在 best 行标出可引用的上限。
  const auto pct = [theoretical](double v) { return 100.0 * v / theoretical; };
  std::printf("\n-- measured streaming ceiling (%zu MiB buffer, same cold/hot discipline as kernels) --\n",
              bench::kStreamBufferBytes >> 20);
  std::printf("%-8s %10s %10s %10s %10s\n", "", "cold GB/s", "% theo", "hot GB/s", "% theo");
  const auto row = [&](const char* name, const bench::StreamResult& r) {
    std::printf("%-8s %10.1f %9.1f%% %10.1f %9.1f%%\n", name, r.cold_gbps(), pct(r.cold_gbps()),
                r.hot_gbps(), pct(r.hot_gbps()));
  };
  row("read", bw.read);
  row("copy", bw.copy);
  row("write", bw.write);
  std::printf("%-8s %10.1f %9.1f%% %10.1f %9.1f%%   <- quote this as the ceiling\n", "best",
              bw.best_cold_gbps(), pct(bw.best_cold_gbps()), bw.best_hot_gbps(),
              pct(bw.best_hot_gbps()));
  std::printf("\nrecord both figures in bench/machine-peaks.json for this machine.\n");

  std::printf(
      "\nnote: peak FLOP/s is not queryable from the runtime; take it from the vendor spec\n"
      "      and record it in docs/environment-matrix.md rather than guessing here.\n");
  return 0;
}
