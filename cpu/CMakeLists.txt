cmake_minimum_required(VERSION 3.28)
# Exploratory CPU benchmark: OpenCV on the CPU vs FKL's CPU backend (BVF), Figure 1 pipeline.
# Meant to be built with the latest stable clang and the best SIMD the CPU has, for example:
#   cmake -S cpu -B build-cpu -DCMAKE_CXX_COMPILER=<path to clang++> -DOpenCV_DIR=<OpenCV build or install>
project(bvf_cpu_bench LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../FusedKernelLibrary" CACHE PATH "FusedKernelLibrary checkout")
# -march=native: every SIMD extension of this CPU (AVX2 and FMA on Zen 3, AVX-512 where available)
set(CPU_ARCH_FLAGS "-march=native" CACHE STRING "CPU code generation flags for the FKL code")

# OpenCVConfig.cmake still calls find_package(CUDA) (OpenCV was built with CUDA), which needs FindCUDA
cmake_policy(SET CMP0146 OLD)
find_package(OpenCV 4.14 REQUIRED COMPONENTS core imgproc)

execute_process(COMMAND git -C ${FKL_DIR} rev-parse --short HEAD OUTPUT_VARIABLE FKL_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${FKL_DIR}/thread-pool rev-parse --short HEAD OUTPUT_VARIABLE THREAD_POOL_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
# The CPU_MULTITHREAD backend uses the thread pool of FKL's thread-pool submodule
find_package(Threads REQUIRED)

add_executable(cpu_bench src/cpu_bench.cpp)
set_target_properties(cpu_bench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(cpu_bench PRIVATE ${FKL_DIR}/include ${FKL_DIR}/thread-pool/include)
separate_arguments(CPU_ARCH_FLAGS_LIST UNIX_COMMAND "${CPU_ARCH_FLAGS}")
target_compile_options(cpu_bench PRIVATE ${CPU_ARCH_FLAGS_LIST} -ftemplate-depth=1000)
target_compile_definitions(cpu_bench PRIVATE BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_THREAD_POOL_COMMIT="${THREAD_POOL_COMMIT}"
                           BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
                           BENCH_CXX_FLAGS="${CMAKE_CXX_FLAGS_RELEASE} -std=c++20 ${CPU_ARCH_FLAGS} -ftemplate-depth=1000")
target_link_libraries(cpu_bench PRIVATE opencv_core opencv_imgproc Threads::Threads)
