add_library(bench_headers INTERFACE)
target_include_directories(bench_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(bench_headers INTERFACE cxx_std_17)

add_executable(bench-probe src/probe_main.cu)
target_link_libraries(bench-probe PRIVATE bench_headers)
