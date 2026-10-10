cmake_minimum_required(VERSION 3.30) # NPPDx requires CMake 3.30
project(FKL-vs-NPPDx LANGUAGES CXX CUDA
        DESCRIPTION "Benchmarks comparing the Fused Kernel Library (FKL) and NVIDIA NPPDx")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Same default as FKL: build for the GPU of this machine
set(CUDA_ARCH "native" CACHE STRING "CUDA architectures to build, passed verbatim to CUDA_ARCHITECTURES")

# NPPDx operations are specialized for one SM version (SM<890> for compute capability 8.9)
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
message(STATUS "NPPDx operations specialized for SM ${NPPDX_SM}")

find_package(CUDAToolkit REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nppdx")
find_package(nppdx CONFIG REQUIRED)

execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/fkl describe --always --tags
                OUTPUT_VARIABLE FKL_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/nppdx describe --always --tags
                OUTPUT_VARIABLE NPPDX_GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

# Same nvcc options FKL uses for its own CUDA targets (fkl/cmake/libs/cuda/target_generation.cmake)
set(BENCHMARK_CUDA_OPTIONS --threads 0 -rdc=false -split-compile=0 -ftemplate-depth=1000 -lineinfo)

enable_testing()

function(add_benchmark TARGET_NAME SOURCE)
    add_executable(${TARGET_NAME} ${SOURCE})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}"
                          CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fkl/include
                                                      ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${BENCHMARK_CUDA_OPTIONS}>)
    target_compile_definitions(${TARGET_NAME} PRIVATE NPPDX_SM=${NPPDX_SM}
                               FKL_VERSION="${FKL_VERSION}" NPPDX_GIT_VERSION="${NPPDX_GIT_VERSION}"
                               CUDA_ARCHITECTURES_STR="${CUDA_ARCH}")
    target_link_libraries(${TARGET_NAME} PRIVATE nppdx::nppdx CUDA::cudart)
    # Like FKL, benchmarks are registered as tests; they write their CSV files to results/ in the build folder
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/results)
endfunction()

add_benchmark(benchmark_figure1_bvf benchmarks/benchmark_figure1_bvf.cu)
