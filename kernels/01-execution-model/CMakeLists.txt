add_executable(bench-probe probe_main.cu)
target_link_libraries(bench-probe PRIVATE bench_headers bench_stream)
