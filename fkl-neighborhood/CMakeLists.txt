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

# Part (b): MorphologyDPP against the libraries, for the erosion (NEIGHBORHOOD_OP 0) and the opening (1)
foreach(OP_INDEX 0 1)
    if(OP_INDEX EQUAL 0)
        set(OP_NAME erode)
    else()
        set(OP_NAME open)
    endif()
    set(TARGET_NAME benchmark_npp_neighborhood_${OP_NAME})
    add_bench_target(${TARGET_NAME} src/benchmark_npp_neighborhood.cu)
    target_compile_definitions(${TARGET_NAME} PRIVATE NEIGHBORHOOD_OP=${OP_INDEX})
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nppc CUDA::nppicc CUDA::nppidei CUDA::nppial CUDA::nppig
                                                 CUDA::nppif CUDA::nppim)
endforeach()

# The OpenCV CUDA programs need OpenCV 4.14 with CUDA and the opencv_contrib modules cudaarithm, cudafilters,
# cudaimgproc and cudawarping (-DOpenCV_DIR=<OpenCV build or install>), and NPP for the NV12 conversion
option(BENCH_OPENCV "Build the OpenCV CUDA programs" ON)
if(BENCH_OPENCV)
    # OpenCVConfig.cmake still calls find_package(CUDA), which needs the legacy FindCUDA module
    cmake_policy(SET CMP0146 OLD)
    find_package(OpenCV 4.14 REQUIRED COMPONENTS core imgproc cudaarithm cudafilters cudaimgproc cudawarping)
    foreach(OP_INDEX 0 1)
        if(OP_INDEX EQUAL 0)
            set(OP_NAME erode)
        else()
            set(OP_NAME open)
        endif()
        set(TARGET_NAME benchmark_opencv_cuda_neighborhood_${OP_NAME})
        add_bench_target(${TARGET_NAME} src/benchmark_opencv_cuda_neighborhood.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE NEIGHBORHOOD_OP=${OP_INDEX})
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nppc CUDA::nppicc opencv_core opencv_imgproc
                                                     opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc
                                                     opencv_cudawarping)
    endforeach()
endif()
