cmake_minimum_required(VERSION 3.28)
# The neighborhood experiment (Experiment 3, see README.md): the Figure 1 pipeline with a morphology stage after the
# resize, FKL against the libraries and FKL's BVF operations against MorphologyDPP:
#   cmake -S fkl-neighborhood -B build-neighborhood
project(fkl_neighborhood LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)

# ErodeBVF and DilateBVF (common/bench_neighborhood_ops.h) against the host reference of common/bench_neighborhood.h
add_bench_target(test_neighborhood_ops src/test_neighborhood_ops.cu)
