cmake_minimum_required(VERSION 3.28)
# Exploratory CPU benchmark: OpenCV on the CPU vs FKL's CPU backend (BVF), Figure 1 pipeline.
# Meant to be built with the latest stable clang and the best SIMD the CPU has, for example:
#   cmake -S cpu -B build-cpu -DCMAKE_CXX_COMPILER=<path to clang++> -DOpenCV_DIR=<OpenCV build or install>
project(bvf_cpu_bench LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fkl" CACHE PATH "FusedKernelLibrary checkout (default: the fkl submodule)")
if(MSVC)
    # cl and clang-cl: AVX2, as FKL's ARCH_FLAGS default on Windows (cl has no equivalent of -march=native)
    set(DEFAULT_CPU_ARCH_FLAGS "/arch:AVX2")
else()
    # -march=native: every SIMD extension of this CPU (AVX2 and FMA on Zen 3, AVX-512 or SVE where available)
    set(DEFAULT_CPU_ARCH_FLAGS "-march=native")
endif()
set(CPU_ARCH_FLAGS "${DEFAULT_CPU_ARCH_FLAGS}" CACHE STRING "CPU code generation flags for the FKL code")
# Timed and warm-up iterations of every benchmark: 1000 and 100 for the paper's results; the CI runs 1 and 0
set(BENCH_ITERATIONS 1000 CACHE STRING "Timed iterations of every benchmark (at least 1)")
set(BENCH_WARMUP_ITERATIONS 100 CACHE STRING "Warm-up iterations before the timed ones")

# OpenCVConfig.cmake still calls find_package(CUDA) when OpenCV was built with CUDA, which needs FindCUDA
cmake_policy(SET CMP0146 OLD)
find_package(OpenCV 4.14 REQUIRED COMPONENTS core imgproc)

execute_process(COMMAND git -C ${FKL_DIR} rev-parse --short HEAD OUTPUT_VARIABLE FKL_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${FKL_DIR}/thread-pool rev-parse --short HEAD OUTPUT_VARIABLE THREAD_POOL_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
# The CPU_MULTITHREAD backend uses the thread pool of FKL's thread-pool submodule
find_package(Threads REQUIRED)

add_executable(cpu_bench src/cpu_bench.cpp)
set_target_properties(cpu_bench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(cpu_bench PRIVATE ${FKL_DIR}/include ${FKL_DIR}/thread-pool/include
                                             ${CMAKE_CURRENT_SOURCE_DIR}/../common)
separate_arguments(CPU_ARCH_FLAGS_LIST NATIVE_COMMAND "${CPU_ARCH_FLAGS}")
if(MSVC)
    # As FKL's tests with cl and clang-cl; cl has no template depth option, clang-cl takes clang's
    set(EXTRA_FLAGS /bigobj /Zc:preprocessor)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND EXTRA_FLAGS /clang:-ftemplate-depth=1000)
    endif()
    set(STD_FLAG /std:c++20)
else()
    set(EXTRA_FLAGS -ftemplate-depth=1000)
    set(STD_FLAG -std=c++20)
endif()
target_compile_options(cpu_bench PRIVATE ${CPU_ARCH_FLAGS_LIST} ${EXTRA_FLAGS})
string(JOIN " " EXTRA_FLAGS_STR ${EXTRA_FLAGS})
# FK_CPU_MULTITHREAD_ENABLED turns on FKL's CPU_MULTITHREAD backend (what FKL::FKL defines when FKL is configured with it)
target_compile_definitions(cpu_bench PRIVATE FK_CPU_MULTITHREAD_ENABLED
                           BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_THREAD_POOL_COMMIT="${THREAD_POOL_COMMIT}"
                           BENCH_ITERATIONS=${BENCH_ITERATIONS} BENCH_WARMUP_ITERATIONS=${BENCH_WARMUP_ITERATIONS}
                           BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
                           BENCH_CXX_FLAGS="${CMAKE_CXX_FLAGS_RELEASE} ${STD_FLAG} ${CPU_ARCH_FLAGS} ${EXTRA_FLAGS_STR}")
target_link_libraries(cpu_bench PRIVATE opencv_core opencv_imgproc Threads::Threads)
