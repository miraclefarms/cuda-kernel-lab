#pragma once

// SMEM staging template — the part post 03 leaves behind for 04.
//
// SMEM staging 模板：第 03 篇留给后续篇目的零件。
//
// 为什么直接写 PTX 指令而不用 cuda::memcpy_async：
//   1. memcpy_async 在 sm_90 上遇到「16 字节对齐 + SMEM 里的 block-scope barrier」会自动改走
//      cp.async.bulk（也就是 TMA），那是第 04 篇的主题。第 03 篇要在三台机器上比同一条
//      指令，只能绕开这个自动分发。
//   2. memcpy_async 配非 SMEM barrier 时，会在函数内部立刻 cp.async.wait_all——搬运发出去
//      马上等，异步的意义就没了。第 03 篇要演示的恰恰是「发起搬运之后先去干别的，晚点再等」，
//      必须自己控制 wait_all 的位置。
// 两条指令的写法与 libcudacxx 的 __cp_async_shared_global<16> / memcpy_completion 保持一致
// （CUDA 13.3.1 头文件），sm_80 起可用，所以 A100 与 H20/H200 跑的是同一条指令。
//
// Why raw PTX instead of cuda::memcpy_async: on sm_90 memcpy_async silently upgrades a
// 16-byte-aligned copy with an SMEM barrier to cp.async.bulk (TMA, post 04), and with
// a non-SMEM barrier it issues cp.async.wait_all immediately, which removes the very
// overlap this post measures. The two instructions below mirror libcudacxx's own
// __cp_async_shared_global<16> and memcpy_completion (CUDA 13.3.1), available from
// sm_80, so all three machines execute the same instruction.

#include <cuda_runtime.h>

namespace staging {

// 16 字节一条的异步搬运：global -> SMEM，跳过 L1（.cg）。调用方负责两件事：
// dst 在 SMEM、src 在 global，且两者都按 16 字节对齐。
// One async 16-byte copy, global -> SMEM, skipping L1. Caller guarantees dst is in
// SMEM, src in global memory, both 16-byte aligned.
__device__ inline void cp_async_16(float* shared_dst, const float* global_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %2;"
               :
               : "r"(static_cast<unsigned int>(::__cvta_generic_to_shared(shared_dst))),
                 "l"(static_cast<unsigned long long>(::__cvta_generic_to_global(global_src))),
                 "n"(16)
               : "memory");
}

// 阻塞到本线程发出的全部 cp.async 都写进 SMEM 为止。只影响本线程，不影响同 warp 的其他线程。
// Block until every cp.async issued by this thread has landed in SMEM.
__device__ inline void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" : : : "memory");
}

// 把一行 global 数据以 16 字节为单位异步搬进 SMEM。count 必须是 4 的倍数——不是的话
// 调用方要走同步路径，这就是「边界仍要手判」。
// Stage one row asynchronously in 16-byte steps. count must be a multiple of 4;
// anything else is the caller's boundary case to handle synchronously.
__device__ inline void cp_async_row(float* shared_dst, const float* global_src, int count) {
  for (int k = 0; k < count; k += 4) cp_async_16(shared_dst + k, global_src + k);
}

}  // namespace staging
