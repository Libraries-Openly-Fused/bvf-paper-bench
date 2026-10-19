cmake_minimum_required(VERSION 3.28)
# FKL vs Halide on the CPU: a Halide pipeline (JIT, compiled once before the timed loops) vs FKL's CPU backends (BVF),
# Figure 1 pipeline (see README.md). Halide is the binary release of Halide 21.0.0 (Halide_DIR: <release>/lib/cmake/Halide),
# for example:
#   cmake -S fkl-vs-halide-cpu -B build-halide-cpu -DCMAKE_CXX_COMPILER=<path to clang++> \
#         -DHalide_DIR=<Halide-21.0.0-x86-64-linux>/lib/cmake/Halide
project(fkl_vs_halide_cpu LANGUAGES CXX)
# FKL, CPU_ARCH_FLAGS, the iterations, the workload and add_cpu_bench_target, shared by every CPU benchmark
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_cpu.cmake)

find_package(Halide 21.0.0 EXACT REQUIRED)
# The release the benchmark was built with, for the platform file: its folder name names the version and the platform
get_filename_component(BENCH_HALIDE_RELEASE "${Halide_DIR}/../../.." ABSOLUTE)
get_filename_component(BENCH_HALIDE_RELEASE "${BENCH_HALIDE_RELEASE}" NAME)

# Halide::Halide: libHalide, with the JIT compiler and its runtime
add_cpu_bench_target(benchmark_halide_cpu src/benchmark_halide_cpu.cpp LIBRARIES Halide::Halide)
target_compile_definitions(benchmark_halide_cpu_code PRIVATE BENCH_HALIDE_RELEASE="${BENCH_HALIDE_RELEASE}")
