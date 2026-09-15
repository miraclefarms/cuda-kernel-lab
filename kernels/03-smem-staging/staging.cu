// 03-smem-staging — who moves the data, and who waits while it moves.
//
// 03-smem-staging：谁来搬数据，搬的时候谁在等。
//
// 同一个工作负载，四种「把 tile 从 global 喂到计算里」的写法：
//
//   global          不搬。计算直接反复读 global（ld.global），靠 L2 兜底。
//   sync-stage      baseline。ld.global + st.shared 把 tile 同步抄进 SMEM，再算。
//                   搬运期间线程什么也干不了。
//   cp-async-wait   发 cp.async 搬 tile，立刻 wait_all，再算。指令是异步的，用法是同步的——
//                   用来把「换了条指令」和「搬运与计算重叠」两个效应拆开。
//   cp-async        optimized。发 cp.async 搬下一个 tile，趁它在飞的时候算当前 tile，
//                   算完再 wait_all。搬运第一次和计算重叠。
//
// 工作负载：global 上一块 bench::kStreamBufferBytes 的 float 数组，视作宽 W 的二维网格，
// 切成 Rows x Cols 的 tile；每个 tile 做 passes 趟依赖链 s = 0.5*s + v[m]，结果写回
// global。依赖链保证每一趟都得重新读数据，编译器无法把多趟合并。
//
// 扫描：tile 形状（决定 SMEM 用量与每 tile 的指令数）× 访问模式（1d 连续 / 2d 分行，2d 下
// 地址要手算、边缘 tile 要手判）。
//
// Tile shapes must be compile-time: SMEM arrays are sized by them. The instantiated
// list is kShapes below; --shapes selects a subset at run time.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/csv.hpp"
#include "bench/stream.cuh"
#include "bench/timing.cuh"
#include "cp_async.cuh"

namespace {

// block 内线程数固定为 256：SMEM 数组的第二维按它开，必须是编译期常量。
constexpr int kBlock = 256;

// 一次启动的全部参数。global 数组共 H*W 个有效元素（尾部不足一行的丢弃），tile 网格
// tile_rows x tile_cols，编号 t = ty * tile_cols + tx。
struct Layout {
  size_t width = 0;      // W：每行元素数；1d 模式下等于元素总数（只有一行）
  size_t height = 0;     // H：完整行数
  size_t tile_rows = 0;  // ceil(H / Rows)
  size_t tile_cols = 0;  // ceil(W / Cols)
  int passes = 4;
};

// 第 t 个 tile 在 global 网格中的位置与实际大小（边缘 tile 会被裁小）。
struct TileGeom {
  size_t y0, x0;
  int rh, cw;
};

__host__ __device__ inline TileGeom tile_geom(size_t t, const Layout& L, int rows, int cols) {
  TileGeom g{};
  const size_t ty = t / L.tile_cols;
  const size_t tx = t % L.tile_cols;
  g.y0 = ty * rows;
  g.x0 = tx * cols;
  g.rh = static_cast<int>(std::min<size_t>(rows, L.height - g.y0));
  g.cw = static_cast<int>(std::min<size_t>(cols, L.width - g.x0));
  return g;
}

// 依赖链：s_m = 0.5 * s_{m-1} + v[m]。0.5*s 精确可表示，所以 host 与 device 按同一顺序
// 算出的结果逐位一致。
__device__ inline float chain(const float* v, int len, int passes) {
  float s = 0.0f;
  for (int p = 0; p < passes; ++p) {
    for (int m = 0; m < len; ++m) s = 0.5f * s + v[m];
  }
  return s;
}

// 同步搬运：ld.global + st.shared，逐元素。边缘 tile 在所有 staging 变体里都走这条路。
__device__ inline void stage_sync(float* sh, const float* src, const TileGeom& g, size_t width) {
  for (int r = 0; r < g.rh; ++r) {
    const float* row = src + (g.y0 + r) * width + g.x0;
    for (int c = 0; c < g.cw; ++c) sh[r * g.cw + c] = row[c];
  }
}

// 只有完整 tile 且列数是 4 的倍数时才能按 16 字节发 cp.async；其余情况调用方走同步。
__device__ inline bool can_cp_async(const TileGeom& g, int rows, int cols) {
  return g.rh == rows && g.cw == cols && (cols % 4) == 0;
}

// 每行一组 cp.async，行首地址由 SM 手算——第 6 节「地址计算仍在 SM」指的就是这里。
__device__ inline void stage_cp_async(float* sh, const float* src, const TileGeom& g, size_t width) {
  for (int r = 0; r < g.rh; ++r) {
    staging::cp_async_row(sh + r * g.cw, src + (g.y0 + r) * width + g.x0, g.cw);
  }
}

// ---------------------------------------------------------------------------
// global：不搬，直接在 global 上跑依赖链。
// ---------------------------------------------------------------------------
template <int Rows, int Cols>
__global__ void k_global(float* dst, const float* src, Layout L, size_t ntiles) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < ntiles; t += stride) {
    const TileGeom g = tile_geom(t, L, Rows, Cols);
    float s = 0.0f;
    for (int p = 0; p < L.passes; ++p) {
      for (int r = 0; r < g.rh; ++r) {
        const float* row = src + (g.y0 + r) * L.width + g.x0;
        for (int c = 0; c < g.cw; ++c) s = 0.5f * s + row[c];
      }
    }
    dst[t] = s;
  }
}

// ---------------------------------------------------------------------------
// sync-stage（baseline）：同步抄进 SMEM，再算。
// ---------------------------------------------------------------------------
template <int Rows, int Cols>
__global__ void k_sync_stage(float* dst, const float* src, Layout L, size_t ntiles) {
  __shared__ __align__(128) float sh[kBlock][Rows * Cols];
  const int tid = threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + tid; t < ntiles; t += stride) {
    const TileGeom g = tile_geom(t, L, Rows, Cols);
    stage_sync(sh[tid], src, g, L.width);
    dst[t] = chain(sh[tid], g.rh * g.cw, L.passes);
  }
}

// ---------------------------------------------------------------------------
// cp-async-wait：异步指令，同步用法。
// ---------------------------------------------------------------------------
template <int Rows, int Cols>
__global__ void k_cp_async_wait(float* dst, const float* src, Layout L, size_t ntiles) {
  __shared__ __align__(128) float sh[kBlock][Rows * Cols];
  const int tid = threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + tid; t < ntiles; t += stride) {
    const TileGeom g = tile_geom(t, L, Rows, Cols);
    if (can_cp_async(g, Rows, Cols)) {
      stage_cp_async(sh[tid], src, g, L.width);
      staging::cp_async_wait_all();
    } else {
      stage_sync(sh[tid], src, g, L.width);
    }
    dst[t] = chain(sh[tid], g.rh * g.cw, L.passes);
  }
}

// ---------------------------------------------------------------------------
// cp-async（optimized）：两个 SMEM 槽轮换。发起「下一个 tile」的搬运 -> 算「当前 tile」
// -> wait_all。代价是 SMEM 用量翻倍。
// Two SMEM slots: issue the next tile's copy, compute the current one while it is in
// flight, then wait. Costs twice the SMEM of the synchronous version.
// ---------------------------------------------------------------------------
template <int Rows, int Cols>
__global__ void k_cp_async(float* dst, const float* src, Layout L, size_t ntiles) {
  __shared__ __align__(128) float sh[2][kBlock][Rows * Cols];
  const int tid = threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + tid;
  if (t >= ntiles) return;

  // 第一个 tile 没有可重叠的前序计算，搬完就等。
  int cur = 0;
  TileGeom g = tile_geom(t, L, Rows, Cols);
  if (can_cp_async(g, Rows, Cols)) {
    stage_cp_async(sh[cur][tid], src, g, L.width);
    staging::cp_async_wait_all();
  } else {
    stage_sync(sh[cur][tid], src, g, L.width);
  }

  for (; t < ntiles; t += stride) {
    const size_t next = t + stride;
    TileGeom gn{};
    if (next < ntiles) {
      gn = tile_geom(next, L, Rows, Cols);
      // 下一个 tile 的搬运发出去，不等。
      if (can_cp_async(gn, Rows, Cols)) {
        stage_cp_async(sh[1 - cur][tid], src, gn, L.width);
      } else {
        stage_sync(sh[1 - cur][tid], src, gn, L.width);
      }
    }
    // 搬运在飞的同时，算当前 tile。
    dst[t] = chain(sh[cur][tid], g.rh * g.cw, L.passes);
    // 算完再等，下一轮读的槽此时已经就绪。
    staging::cp_async_wait_all();
    cur = 1 - cur;
    g = gn;
  }
}

// ---------------------------------------------------------------------------
// host 侧
// ---------------------------------------------------------------------------

struct Shape {
  int rows;
  int cols;
};

// 编译期实例化的 tile 形状，SMEM 用量 = 256 线程 x 元素数 x 4 B（cp-async 再 x2 槽）。
//
// 实测的 SMEM 上限是按编译目标卡死的：ptxas 对 sm_80 目标限每个入口函数 0xc000
// （48 KiB），sm_90a 目标能放下 128 KiB。16 元素的 tile（cp-async 两槽 32 KiB）三台机器
// 都能编；64 元素的 tile（sync 64 KiB、cp-async 128 KiB）只在 sm_90a 上存在——A100 上它
// 根本编不过，这本身就是第 1 节「每层的容量与代价」的一条数据。
//
// SMEM limits are enforced per compile target: ptxas caps an sm_80 entry function at
// 48 KiB, sm_90a accepts 128 KiB. 16-element tiles build everywhere; 64-element tiles
// exist only on sm_90a builds.
#define STAGING_SHAPES_COMMON(X) X(1, 4) X(1, 16) X(4, 4) X(2, 8)
// STAGING_LARGE_TILES 由 CMake 按 CMAKE_CUDA_ARCHITECTURES 定义：__CUDA_ARCH__ 对 host 代码不可见，
// 而形状表与分发都在 host 侧。
#if defined(STAGING_LARGE_TILES)
#define STAGING_SHAPES_LARGE(X) X(1, 64) X(4, 16) X(8, 8)
#else
#define STAGING_SHAPES_LARGE(X)
#endif
#define STAGING_SHAPE_ENTRY(R, C) {R, C},
const Shape kShapes[] = {STAGING_SHAPES_COMMON(STAGING_SHAPE_ENTRY)
                             STAGING_SHAPES_LARGE(STAGING_SHAPE_ENTRY)};
#undef STAGING_SHAPE_ENTRY

enum class Variant { kGlobal, kSyncStage, kCpAsyncWait, kCpAsync };

void launch(Variant v, const Shape& sh, int grid, cudaStream_t st, float* dst, const float* src,
            const Layout& L, size_t ntiles) {
  // SMEM 数组维度必须编译期确定，只能按形状显式分发。
#define STAGING_DISPATCH(R, C)                                                            \
  if (sh.rows == R && sh.cols == C) {                                                     \
    switch (v) {                                                                          \
      case Variant::kGlobal:                                                              \
        k_global<R, C><<<grid, kBlock, 0, st>>>(dst, src, L, ntiles);                     \
        return;                                                                           \
      case Variant::kSyncStage:                                                           \
        k_sync_stage<R, C><<<grid, kBlock, 0, st>>>(dst, src, L, ntiles);                 \
        return;                                                                           \
      case Variant::kCpAsyncWait:                                                         \
        k_cp_async_wait<R, C><<<grid, kBlock, 0, st>>>(dst, src, L, ntiles);              \
        return;                                                                           \
      case Variant::kCpAsync:                                                             \
        k_cp_async<R, C><<<grid, kBlock, 0, st>>>(dst, src, L, ntiles);                   \
        return;                                                                           \
    }                                                                                     \
  }
  STAGING_SHAPES_COMMON(STAGING_DISPATCH)
  STAGING_SHAPES_LARGE(STAGING_DISPATCH)
#undef STAGING_DISPATCH
  std::fprintf(stderr, "shape %dx%d is not instantiated for this architecture\n", sh.rows, sh.cols);
  std::exit(1);
}

// host 黄金结果：与 device 完全相同的遍历顺序与运算，结果应逐位一致。
std::vector<float> golden(const std::vector<float>& src, const Layout& L, const Shape& sh,
                          size_t ntiles) {
  std::vector<float> out(ntiles);
  for (size_t t = 0; t < ntiles; ++t) {
    const TileGeom g = tile_geom(t, L, sh.rows, sh.cols);
    float s = 0.0f;
    for (int p = 0; p < L.passes; ++p) {
      for (int r = 0; r < g.rh; ++r) {
        const float* row = src.data() + (g.y0 + r) * L.width + g.x0;
        for (int c = 0; c < g.cw; ++c) s = 0.5f * s + row[c];
      }
    }
    out[t] = s;
  }
  return out;
}

bool verify(const std::vector<float>& got, const std::vector<float>& want, const char* variant) {
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "[verify] %s mismatch at tile %zu: got %.9g want %.9g\n", variant, i,
                   got[i], want[i]);
      return false;
    }
  }
  return true;
}

std::vector<Shape> parse_shapes(const std::string& list) {
  std::vector<Shape> out;
  size_t pos = 0;
  while (pos < list.size()) {
    const size_t end = std::min(list.find(',', pos), list.size());
    const std::string tok = list.substr(pos, end - pos);
    const size_t x = tok.find('x');
    if (x == std::string::npos) {
      std::fprintf(stderr, "bad shape '%s', expected RxC\n", tok.c_str());
      std::exit(1);
    }
    out.push_back({std::atoi(tok.substr(0, x).c_str()), std::atoi(tok.substr(x + 1).c_str())});
    pos = end + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  int warmup = 5;
  int samples = 30;
  int batch = 10;
  int passes = 4;
  // 2d 模式的行宽。刻意取 4 的倍数（cp.async 需要 16 字节对齐）但不是 8 的倍数：
  // 列数 >= 8 的 tile 在每行末尾都会被裁小，边界处理的代价因此一定出现在数据里。
  // 2d row width: a multiple of 4 (16-byte alignment) but not of 8, so tiles with
  // cols >= 8 are clipped at every row end and the boundary path always shows up.
  size_t width_2d = 4092;
  std::string shapes_arg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
    else if (arg == "--batch" && i + 1 < argc) batch = std::atoi(argv[++i]);
    else if (arg == "--passes" && i + 1 < argc) passes = std::atoi(argv[++i]);
    else if (arg == "--width" && i + 1 < argc) width_2d = std::strtoull(argv[++i], nullptr, 10);
    else if (arg == "--shapes" && i + 1 < argc) shapes_arg = argv[++i];
  }
  if (width_2d % 4 != 0) {
    std::fprintf(stderr, "--width must be a multiple of 4 (16-byte alignment for cp.async)\n");
    return 1;
  }
  const std::vector<Shape> shapes =
      shapes_arg.empty() ? std::vector<Shape>(std::begin(kShapes), std::end(kShapes))
                         : parse_shapes(shapes_arg);

  // 工作集与 bench-probe 的 streaming 上限相同，带宽数字才可比。
  const size_t bytes = bench::kStreamBufferBytes;
  const size_t n = bytes / sizeof(float);

  std::vector<float> hsrc(n);
  for (size_t i = 0; i < n; ++i) hsrc[i] = static_cast<float>(i % 13);

  float* src = nullptr;
  float* dst = nullptr;
  BENCH_CHECK(cudaMalloc(&src, bytes));
  BENCH_CHECK(cudaMemcpy(src, hsrc.data(), bytes, cudaMemcpyHostToDevice));

  cudaDeviceProp prop{};
  BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount * 32;

  cudaStream_t stream{};
  BENCH_CHECK(cudaStreamCreate(&stream));

  const struct {
    Variant v;
    const char* name;
  } variants[] = {{Variant::kGlobal, "global"},
                  {Variant::kSyncStage, "sync-stage"},
                  {Variant::kCpAsyncWait, "cp-async-wait"},
                  {Variant::kCpAsync, "cp-async"}};

  bench::print_csv_header();
  for (const Shape& sh : shapes) {
    Layout L;
    L.passes = passes;
    // 1 行的 tile 视为 1d：整个数组是一行，不存在行边界，地址也不用按行换算。
    const bool is_1d = (sh.rows == 1);
    L.width = is_1d ? n : width_2d;
    L.height = n / L.width;
    L.tile_rows = (L.height + sh.rows - 1) / sh.rows;
    L.tile_cols = (L.width + sh.cols - 1) / sh.cols;
    const size_t ntiles = L.tile_rows * L.tile_cols;
    const size_t elems = L.height * L.width;

    BENCH_CHECK(cudaMalloc(&dst, ntiles * sizeof(float)));
    const std::vector<float> want = golden(hsrc, L, sh, ntiles);
    std::vector<float> got(ntiles);

    char shape[128];
    std::snprintf(shape, sizeof(shape), "pattern=%s;tile=%dx%d;passes=%d;width=%zu",
                  is_1d ? "1d" : "2d", sh.rows, sh.cols, passes, L.width);

    for (const auto& var : variants) {
      auto body = [&](cudaStream_t st) { launch(var.v, sh, grid, st, dst, src, L, ntiles); };

      // Correctness before timing, for every variant at every shape.
      // 每个形状、每个变体都先校验再计时。
      BENCH_CHECK(cudaMemset(dst, 0, ntiles * sizeof(float)));
      body(stream);
      BENCH_CHECK(cudaStreamSynchronize(stream));
      BENCH_CHECK(cudaMemcpy(got.data(), dst, ntiles * sizeof(float), cudaMemcpyDeviceToHost));
      if (!verify(got, want, var.name)) return 1;

      bench::RunRow row;
      row.kernel = "03-smem-staging";
      row.variant = var.name;
      row.shape = shape;
      row.dtype = "fp32";
      // 流量按逻辑工作集计：输入读一遍 + 每 tile 写回一个结果。global 变体物理上会读
      // passes 遍，但那几遍由 L2 承担，按逻辑工作集计才能在变体之间直接比吞吐。
      row.bytes = static_cast<double>(elems) * sizeof(float) +
                  static_cast<double>(ntiles) * sizeof(float);
      row.flops = static_cast<double>(elems) * passes * 2.0;

      row.mode = "cold";
      row.stats = bench::measure_cold(body, warmup, samples, stream);
      bench::print_csv_row(row);

      row.mode = "hot";
      row.stats = bench::measure_hot(body, warmup, samples, batch, stream);
      bench::print_csv_row(row);
    }
    BENCH_CHECK(cudaFree(dst));
    dst = nullptr;
  }

  BENCH_CHECK(cudaStreamDestroy(stream));
  BENCH_CHECK(cudaFree(src));
  return 0;
}
