cmake_minimum_required(VERSION 3.28)
# CPU benchmark of the Figure 1 pipeline: RPP's HOST backend vs FKL's CPU backends (BVF), the ROCm counterpart of
# fkl-vs-opencv-cpu/ (which compares with OpenCV). Meant to be built with a recent clang and the best SIMD of the CPU, for example:
#   cmake -S rocm/cpu -B build-rocm-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-23 \
#         -DRPP_ROOT=<prefix with RPP>   (FKL: the repository's fkl submodule with its thread-pool, or -DFKL_DIR=<checkout>)
project(bvf_rpp_cpu_bench LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# The workload of every benchmark (common/bench_geometry.h): the option BENCH_REGIONS (region sizes: mixed, the paper's,
# or one of 20x10, 32x16, 48x24) and BENCH_GEOMETRY_DEFINITIONS
include(${CMAKE_CURRENT_SOURCE_DIR}/../../common/bench_geometry.cmake)
# Timed and warm-up iterations, as in every benchmark of the repository (common/bench_gpu.cmake)
set(BENCH_ITERATIONS 1000 CACHE STRING "Timed iterations of every benchmark (at least 1)")
set(BENCH_WARMUP_ITERATIONS 100 CACHE STRING "Warm-up iterations before the timed ones")

set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../fkl" CACHE PATH "FusedKernelLibrary checkout (default: the repository's fkl submodule)")
set(ROCM_PATH "/opt/rocm/core-10.1" CACHE PATH "ROCm installation (HIP headers, which rpp.h includes)")
set(RPP_ROOT "${ROCM_PATH}" CACHE PATH "Prefix with include/rpp/rpp.h and lib/librpp.so")
# -march=native: every SIMD extension of this CPU (AVX2 and FMA on Zen 3, AVX-512 where available)
set(CPU_ARCH_FLAGS "-march=native" CACHE STRING "CPU code generation flags for the FKL code")

find_path(RPP_INCLUDE_DIR rpp/rpp.h HINTS "${RPP_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(RPP_LIBRARY rpp HINTS "${RPP_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

# --dirty: a modified FKL tree shows up in cpu_platform.txt
execute_process(COMMAND git -C ${FKL_DIR} describe --always --dirty --abbrev=7 OUTPUT_VARIABLE FKL_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
# The paper's FKL commit: the commit recorded for the repository's fkl submodule
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/../.. ls-tree HEAD fkl OUTPUT_VARIABLE FKL_SUBMODULE_ENTRY
                OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]" FKL_SUBMODULE_COMMIT
       "${FKL_SUBMODULE_ENTRY}")
if(NOT FKL_SUBMODULE_COMMIT OR NOT FKL_COMMIT MATCHES "^${FKL_SUBMODULE_COMMIT}")
    message(WARNING "FKL_DIR (${FKL_DIR}) is at ${FKL_COMMIT}, not ${FKL_SUBMODULE_COMMIT} (the repository's fkl "
                    "submodule, the paper's FKL commit)")
endif()
execute_process(COMMAND git -C ${FKL_DIR}/thread-pool rev-parse --short HEAD OUTPUT_VARIABLE THREAD_POOL_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT EXISTS "${FKL_DIR}/thread-pool/include")
    message(FATAL_ERROR "${FKL_DIR}/thread-pool is empty: the CPU_MULTITHREAD backend needs FKL's thread-pool submodule "
                        "(git submodule update --init --recursive)")
endif()
# The commit of this repository, and whether its checkout has changes, as in rocm/CMakeLists.txt: both show up in
# cpu_platform.txt
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/../.. rev-parse --short HEAD OUTPUT_VARIABLE BENCH_REPO_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/../.. status --porcelain --untracked-files=no
                OUTPUT_VARIABLE BENCH_REPO_CHANGES OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(BENCH_REPO_CHANGES)
    string(APPEND BENCH_REPO_COMMIT " (with uncommitted changes)")
endif()
find_package(Threads REQUIRED)

add_executable(rpp_cpu_bench src/rpp_cpu_bench.cpp)
set_target_properties(rpp_cpu_bench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                                               BUILD_RPATH "${RPP_ROOT}/lib")
target_include_directories(rpp_cpu_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common ${FKL_DIR}/include ${FKL_DIR}/thread-pool/include ${RPP_INCLUDE_DIR}
                                                 ${ROCM_PATH}/include)
separate_arguments(CPU_ARCH_FLAGS_LIST UNIX_COMMAND "${CPU_ARCH_FLAGS}")
target_compile_options(rpp_cpu_bench PRIVATE ${CPU_ARCH_FLAGS_LIST} -ftemplate-depth=1000)
# FK_CPU_MULTITHREAD_ENABLED turns on FKL's CPU_MULTITHREAD backend; __HIP_PLATFORM_AMD__ for the HIP headers rpp.h includes
target_compile_definitions(rpp_cpu_bench PRIVATE ${BENCH_GEOMETRY_DEFINITIONS} FK_CPU_MULTITHREAD_ENABLED __HIP_PLATFORM_AMD__
                           BENCH_ITERATIONS=${BENCH_ITERATIONS} BENCH_WARMUP_ITERATIONS=${BENCH_WARMUP_ITERATIONS}
                           BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_THREAD_POOL_COMMIT="${THREAD_POOL_COMMIT}"
                           BENCH_REPO_COMMIT="${BENCH_REPO_COMMIT}"
                           BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
                           BENCH_CXX_FLAGS="${CMAKE_CXX_FLAGS_RELEASE} -std=c++20 ${CPU_ARCH_FLAGS} -ftemplate-depth=1000"
                           BENCH_RPP_LIB="${RPP_LIBRARY}")
target_link_libraries(rpp_cpu_bench PRIVATE ${RPP_LIBRARY} Threads::Threads)
