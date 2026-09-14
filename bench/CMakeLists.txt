add_library(bench_headers INTERFACE)
target_include_directories(bench_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(bench_headers INTERFACE cxx_std_17)

add_library(bench_stream STATIC src/stream.cu)
target_link_libraries(bench_stream PUBLIC bench_headers)
set_target_properties(bench_stream PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(bench-probe src/probe_main.cu)
target_link_libraries(bench-probe PRIVATE bench_headers bench_stream)
