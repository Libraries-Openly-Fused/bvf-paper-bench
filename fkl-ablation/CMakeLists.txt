cmake_minimum_required(VERSION 3.28)
# FKL's ablation on the Figure 1 pipeline, FKL VF+HF vs FKL BVF and FKL's host overhead, and the C++ type of the fused
# read operation (see README.md):
#   cmake -S fkl-ablation -B build-ablation
project(fkl_ablation LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)

add_bench_target(benchmark_fkl_ablation src/benchmark_fkl_ablation.cu)
add_bench_target(bvf_types src/bvf_types.cu)
