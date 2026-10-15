cmake_minimum_required(VERSION 3.28)
# The neighborhood experiment (Experiment 3, see README.md): 64x32 crops of the frame, a box blur and the
# normalization, FKL's BVF box operation against MorphologyDPP, and MorphologyDPP against the libraries:
#   cmake -S fkl-neighborhood -B build-neighborhood -DOpenCV_DIR=<OpenCV build or install>
# The crops have a fixed size (common/bench_neighborhood.h), so every program, the NPPDx ones too, is built once, and
# BENCH_REGIONS (the sizes of the other experiments) does not change them.
project(fkl_neighborhood LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_gpu.cmake)

# The OpenCV CUDA program needs OpenCV 4.14 with CUDA and the opencv_contrib modules cudaarithm, cudafilters and
# cudaimgproc (-DOpenCV_DIR=<OpenCV build or install>), and NPP for the NV12 conversion
option(BENCH_OPENCV "Build the OpenCV CUDA program" ON)
# The NPPDx programs need CMake 3.30 or newer and the nppdx submodule (or NPPDX_DIR)
option(BENCH_NPPDX "Build the NPPDx programs" ON)

# BoxBVF and MorphologyDPP's box blur against the host reference of common/bench_neighborhood.h
add_bench_target(test_neighborhood_ops src/test_neighborhood_ops.cu)

# Part (a): one program per box size and number of passes (PART_A_KS, PART_A_PASSES of common/bench_neighborhood.h),
# on the crops, and the same with the box after Figure 1's resize (_resize, on Figure 1's regions)
foreach(K 3 5 7)
    foreach(PASSES 1 2)
        foreach(RESIZE 0 1)
            set(TARGET_NAME benchmark_fkl_neighborhood_k${K}_p${PASSES})
            if(RESIZE)
                string(APPEND TARGET_NAME _resize)
            endif()
            add_bench_target(${TARGET_NAME} src/benchmark_fkl_neighborhood.cu)
            target_compile_definitions(${TARGET_NAME} PRIVATE NEIGHBORHOOD_K=${K} NEIGHBORHOOD_PASSES=${PASSES}
                                                              NEIGHBORHOOD_RESIZE=${RESIZE})
        endforeach()
    endforeach()
endforeach()

# Part (b): MorphologyDPP against NPP
add_bench_target(benchmark_npp_neighborhood src/benchmark_npp_neighborhood.cu)
target_link_libraries(benchmark_npp_neighborhood PRIVATE CUDA::nppc CUDA::nppicc CUDA::nppidei CUDA::nppial
                                                         CUDA::nppif)

# Part (b): MorphologyDPP against OpenCV CUDA
if(BENCH_OPENCV)
    # OpenCVConfig.cmake still calls find_package(CUDA), which needs the legacy FindCUDA module
    cmake_policy(SET CMP0146 OLD)
    find_package(OpenCV 4.14 REQUIRED COMPONENTS core imgproc cudaarithm cudafilters cudaimgproc)
    add_bench_target(benchmark_opencv_cuda_neighborhood src/benchmark_opencv_cuda_neighborhood.cu)
    target_link_libraries(benchmark_opencv_cuda_neighborhood PRIVATE CUDA::nppc CUDA::nppicc opencv_core
                                                                     opencv_imgproc opencv_cudaarithm
                                                                     opencv_cudafilters opencv_cudaimgproc)
endif()

# Part (b): MorphologyDPP against NPPDx (hand written normalization and float write: benchmark_nppdx_neighborhood),
# with NPPDx (the nppdx submodule) specialized for one SM version, as in fkl-vs-nppdx
if(BENCH_NPPDX)
    if(CMAKE_VERSION VERSION_LESS 3.30)
        message(FATAL_ERROR "NPPDx requires CMake 3.30 or newer (or configure with -DBENCH_NPPDX=OFF)")
    endif()
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
    set(NPPDX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nppdx" CACHE PATH "NPPDx checkout (default: the nppdx submodule)")
    list(APPEND CMAKE_PREFIX_PATH "${NPPDX_DIR}")
    find_package(nppdx CONFIG REQUIRED)
    execute_process(COMMAND git -C ${NPPDX_DIR} describe --always --tags
                    OUTPUT_VARIABLE NPPDX_GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(WIN32)
        # As fkl-vs-nppdx: the real value of __cplusplus (NPPDx checks for C++17) and no min/max macros
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
        add_compile_definitions(NOMINMAX)
    endif()
    add_bench_target(benchmark_nppdx_neighborhood src/benchmark_nppdx_neighborhood.cu)
    target_compile_definitions(benchmark_nppdx_neighborhood PRIVATE NPPDX_SM=${NPPDX_SM}
                                                                    NPPDX_GIT_VERSION="${NPPDX_GIT_VERSION}")
    target_link_libraries(benchmark_nppdx_neighborhood PRIVATE nppdx::nppdx CUDA::nppc CUDA::nppicc CUDA::nppidei
                                                               CUDA::nppial)
endif()
