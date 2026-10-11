cmake_minimum_required(VERSION 3.28)
project(bvf_paper_bench LANGUAGES CXX CUDA)

# Same default as FKL: the GPU of this machine (sm_89 on the RTX 4090 of the paper)
set(CUDA_ARCH "native" CACHE STRING "CUDA architectures to build, passed verbatim to CUDA_ARCHITECTURES")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
if(CUDA_ARCH STREQUAL "native")
    set(CUDA_ARCH_DESCRIPTION "native (${CMAKE_CUDA_ARCHITECTURES_NATIVE})")
else()
    set(CUDA_ARCH_DESCRIPTION "${CUDA_ARCH}")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fkl" CACHE PATH "FusedKernelLibrary checkout (default: the fkl submodule)")
find_package(CUDAToolkit REQUIRED)
# OpenCVConfig.cmake still calls find_package(CUDA), which needs the legacy FindCUDA module
cmake_policy(SET CMP0146 OLD)
find_package(OpenCV 4.14 REQUIRED COMPONENTS core cudaarithm cudawarping)

execute_process(COMMAND git -C ${FKL_DIR} rev-parse --short HEAD OUTPUT_VARIABLE FKL_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${FKL_DIR} rev-parse --abbrev-ref HEAD OUTPUT_VARIABLE FKL_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)

# Same nvcc options FKL uses for its own CUDA targets (cmake/libs/cuda/target_generation.cmake)
set(BENCH_CUDA_OPTIONS --threads 0 -rdc=false -split-compile=0 -ftemplate-depth=1000 -lineinfo)
if(WIN32)
    # As FKL on Windows: big object files and the conforming preprocessor of MSVC
    list(APPEND BENCH_CUDA_OPTIONS -Xcompiler=/bigobj -Xcompiler=/Zc:preprocessor)
endif()
string(JOIN " " BENCH_CUDA_OPTIONS_STR ${BENCH_CUDA_OPTIONS})
set(BENCH_NVCC_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELEASE} -std=c++20 ${BENCH_CUDA_OPTIONS_STR}, CUDA_ARCHITECTURES ${CUDA_ARCH_DESCRIPTION}")

function(add_bench_target TARGET_NAME SOURCE)
    add_executable(${TARGET_NAME} ${SOURCE})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                                    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_include_directories(${TARGET_NAME} PRIVATE ${FKL_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_OPTIONS}>)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_FKL_BRANCH="${FKL_BRANCH}"
        BENCH_NVCC_VERSION="${CMAKE_CUDA_COMPILER_VERSION}" BENCH_NVCC_FLAGS="${BENCH_NVCC_FLAGS}")
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart CUDA::nppc CUDA::nppicc CUDA::nppig CUDA::nppial
                                                 CUDA::nppidei CUDA::cupti CUDA::nvtx3 opencv_core opencv_cudaarithm opencv_cudawarping)
endfunction()

add_bench_target(bvf_bench src/bvf_bench.cu)
add_bench_target(bvf_types src/bvf_types.cu)
