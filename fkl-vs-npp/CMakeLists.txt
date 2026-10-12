cmake_minimum_required(VERSION 3.28)
# FKL vs NPP on the Figure 1 pipeline (see README.md):
#   cmake -S fkl-vs-npp -B build-npp
project(fkl_vs_npp LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)

add_bench_target(benchmark_npp src/benchmark_npp.cu)
target_link_libraries(benchmark_npp PRIVATE CUDA::nppc CUDA::nppicc CUDA::nppig CUDA::nppial CUDA::nppidei)
