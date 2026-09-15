cmake_minimum_required(VERSION 3.24)
project(cuda-kernel-lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pass explicitly per machine: 80 for A100, 90a for H20 / H200.
# The `a` suffix matters — wgmma, TMA and cluster features are arch-conditional
# and silently unavailable without it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR
    "Set -DCMAKE_CUDA_ARCHITECTURES explicitly (80 for A100, 90a for H20/H200). "
    "Defaulting it would hide which architecture a published number came from.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_subdirectory(bench)
add_subdirectory(kernels/00-template)
add_subdirectory(kernels/01-execution-model)
add_subdirectory(kernels/02-measurement-discipline)
add_subdirectory(kernels/03-smem-staging)
