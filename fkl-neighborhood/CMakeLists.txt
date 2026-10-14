cmake_minimum_required(VERSION 3.28)
# The neighborhood experiment (Experiment 3, see README.md): the Figure 1 pipeline with a morphology stage after the
# resize, FKL's BVF morphology operations against MorphologyDPP, and MorphologyDPP against the libraries:
#   cmake -S fkl-neighborhood -B build-neighborhood
project(fkl_neighborhood LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)

# ErodeBVF and DilateBVF (common/bench_neighborhood_ops.h) against the host reference of common/bench_neighborhood.h
add_bench_target(test_neighborhood_ops src/test_neighborhood_ops.cu)

# Part (a): one program per mask size and number of passes (PART_A_KS, PART_A_PASSES of common/bench_neighborhood.h)
foreach(K 3 5 7)
    foreach(PASSES 1 2)
        set(TARGET_NAME benchmark_fkl_neighborhood_k${K}_p${PASSES})
        add_bench_target(${TARGET_NAME} src/benchmark_fkl_neighborhood.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE NEIGHBORHOOD_K=${K} NEIGHBORHOOD_PASSES=${PASSES})
    endforeach()
endforeach()
