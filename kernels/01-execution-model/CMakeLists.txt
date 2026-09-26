add_executable(bench-probe probe_main.cu)
target_link_libraries(bench-probe PRIVATE bench_headers bench_stream)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(OCCUPANCY_PROBES "${CMAKE_CURRENT_BINARY_DIR}/occupancy_probes.cuh")
add_custom_command(
  OUTPUT "${OCCUPANCY_PROBES}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_occupancy_probes.py"
          "${OCCUPANCY_PROBES}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_occupancy_probes.py"
  VERBATIM)
add_executable(occupancy-scan occupancy_scan.cu "${OCCUPANCY_PROBES}")
target_include_directories(occupancy-scan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
