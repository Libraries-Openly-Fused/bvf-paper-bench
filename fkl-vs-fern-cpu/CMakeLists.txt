cmake_minimum_required(VERSION 3.28)
# FKL vs Fern on the CPU: the code Fern (gern) generates for the Figure 1 pipeline vs FKL's CPU backends (BVF), see
# README.md. The generated code is checked in (fern/generated/); building the benchmark needs neither gern nor its
# dependencies (fern/regenerate.cmake rebuilds the generated code from the gern submodule). For example:
#   cmake -S fkl-vs-fern-cpu -B build-fern-cpu -DCMAKE_CXX_COMPILER=<path to clang++>
project(fkl_vs_fern_cpu LANGUAGES CXX)
# FKL, CPU_ARCH_FLAGS, the iterations, the workload and add_cpu_bench_target, shared by every CPU benchmark
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_cpu.cmake)

# The gern commit the code was generated with: the repository's gern submodule
set(GERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gern" CACHE PATH "gern checkout the generated code comes from")
execute_process(COMMAND git -C ${GERN_DIR} rev-parse --short HEAD OUTPUT_VARIABLE GERN_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT GERN_COMMIT)
    # Without the submodule checked out, the commit the repository records for it
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/.. ls-tree HEAD gern OUTPUT_VARIABLE GERN_ENTRY
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX MATCH "[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]" GERN_COMMIT "${GERN_ENTRY}")
endif()

# The generated code and the black boxes are compiled with the flags of FKL's code (CPU_ARCH_FLAGS), like the harness
add_cpu_bench_target(benchmark_fern_cpu src/benchmark_fern_cpu.cpp fern/generated/fern_band16.cpp)
target_include_directories(benchmark_fern_cpu_code PRIVATE fern)
target_compile_definitions(benchmark_fern_cpu_code PRIVATE BENCH_GERN_COMMIT="${GERN_COMMIT}")
