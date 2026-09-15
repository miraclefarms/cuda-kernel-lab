add_executable(kernel-03-smem-staging staging.cu)
target_link_libraries(kernel-03-smem-staging PRIVATE bench_headers)

# 64-element tiles exceed the sm_80 SMEM cap (48 KiB per entry function), so they are
# only built when no targeted architecture is older than sm_90. See staging.cu.
# 64 元素的 tile 超过 sm_80 的 SMEM 上限，只在全部目标架构 >= sm_90 时编译。
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)[5-8][0-9]")
  target_compile_definitions(kernel-03-smem-staging PRIVATE STAGING_LARGE_TILES=1)
endif()
