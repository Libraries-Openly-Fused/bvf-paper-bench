cmake_minimum_required(VERSION 3.30) # NPPDx requires CMake 3.30
# nppdx-wins (src/wins_common.h): a smoke exploration of a 5x5 box BEFORE a resize with a compile-time ratio, the order
# NPPDx is built for, against FKL's two forms (inlined BoxBVF, and MorphologyDPP into an intermediate + resize):
#   cmake -S nppdx-wins -B build-wins -DBENCH_ITERATIONS=10 -DBENCH_WARMUP_ITERATIONS=3
# W1 uses Experiment 1's positions with every region of 32x16, so BENCH_REGIONS is 32x16.
set(BENCH_REGIONS 32x16 CACHE STRING "Region sizes (nppdx-wins: 32x16)")
project(nppdx_wins LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)
if(NOT BENCH_REGIONS STREQUAL "32x16")
    message(FATAL_ERROR "nppdx-wins is built for 32x16 regions, not ${BENCH_REGIONS}")
endif()

# NPPDx (the nppdx submodule), its operations specialized for one SM version, as in fkl-neighborhood
execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE DETECTED_COMPUTE_CAP OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" DETECTED_COMPUTE_CAP "${DETECTED_COMPUTE_CAP}")
if(DETECTED_COMPUTE_CAP)
    string(REPLACE "." "" DEFAULT_NPPDX_SM "${DETECTED_COMPUTE_CAP}")
    string(APPEND DEFAULT_NPPDX_SM "0")
else()
    set(DEFAULT_NPPDX_SM 890)
endif()
set(NPPDX_SM ${DEFAULT_NPPDX_SM} CACHE STRING "SM version used for the NPPDx operations, for example 890")
set(NPPDX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nppdx" CACHE PATH "NPPDx checkout (default: the nppdx submodule)")
list(APPEND CMAKE_PREFIX_PATH "${NPPDX_DIR}")
find_package(nppdx CONFIG REQUIRED)
execute_process(COMMAND git -C ${NPPDX_DIR} describe --always --tags
                OUTPUT_VARIABLE NPPDX_GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

foreach(PROGRAM regions frame)
    add_bench_target(benchmark_wins_${PROGRAM} src/benchmark_wins_${PROGRAM}.cu)
    target_compile_definitions(benchmark_wins_${PROGRAM} PRIVATE NPPDX_SM=${NPPDX_SM}
                                                                 NPPDX_GIT_VERSION="${NPPDX_GIT_VERSION}")
    target_link_libraries(benchmark_wins_${PROGRAM} PRIVATE nppdx::nppdx CUDA::nppc CUDA::nppicc CUDA::nppidei
                                                            CUDA::nppial)
endforeach()
