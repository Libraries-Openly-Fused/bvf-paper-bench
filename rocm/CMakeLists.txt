cmake_minimum_required(VERSION 3.28)
# ROCm port of the GPU benchmark (src/bvf_bench.cu): FKL's HIP backend against RPP, on an AMD GPU.
# Configure with amdclang++ as both the C++ and the HIP compiler, as FKL's ROCm CI does, for example:
#   cmake -S rocm -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
#         -DRPP_ROOT=<prefix with include/rpp and lib/librpp.so> -DROCPROFILER_SDK_ROOT=<prefix with rocprofiler-sdk>
project(bvf_paper_bench_rocm LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Radeon RX 9060 only (gfx1200), FKL's default HIP architecture (cmake/libs/hip/archs.cmake)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1200)
endif()

set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../FusedKernelLibrary" CACHE PATH "FusedKernelLibrary checkout (branch bvf-paper-bench, b9f4281)")
set(ROCM_PATH "/opt/rocm/core-10.1" CACHE PATH "ROCm installation")
set(RPP_ROOT "${ROCM_PATH}" CACHE PATH "Prefix with include/rpp/rpp.h and lib/librpp.so")
set(ROCPROFILER_SDK_ROOT "${ROCM_PATH}" CACHE PATH "Prefix with include/rocprofiler-sdk and lib/librocprofiler-sdk.so")

find_path(RPP_INCLUDE_DIR rpp/rpp.h HINTS "${RPP_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(RPP_LIBRARY rpp HINTS "${RPP_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_path(ROCPROFILER_SDK_INCLUDE_DIR rocprofiler-sdk/rocprofiler.h HINTS "${ROCPROFILER_SDK_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ROCPROFILER_SDK_LIBRARY rocprofiler-sdk HINTS "${ROCPROFILER_SDK_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

# --dirty: a modified FKL tree shows up in platform.txt
execute_process(COMMAND git -C ${FKL_DIR} describe --always --dirty --abbrev=7 OUTPUT_VARIABLE FKL_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT FKL_COMMIT MATCHES "^b9f4281")
    message(WARNING "FKL_DIR (${FKL_DIR}) is at ${FKL_COMMIT}, not b9f4281 (the paper's GPU commit)")
endif()
execute_process(COMMAND git -C ${FKL_DIR} rev-parse --abbrev-ref HEAD OUTPUT_VARIABLE FKL_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)

# FKL's own HIP target options (cmake/libs/hip/target_generation.cmake, debug.cmake): HIP_STANDARD 20 (required),
# -ftemplate-depth=1000 (TEMPLATE_DEPTH), -gline-tables-only (ENABLE_HIP_LINE_INFO); Release adds -O3 -DNDEBUG
set(BENCH_HIP_OPTIONS -ftemplate-depth=1000 -gline-tables-only)
string(JOIN " " BENCH_HIP_OPTIONS_STR ${BENCH_HIP_OPTIONS})
set(BENCH_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_RELEASE} -std=gnu++20 --offload-arch=${CMAKE_HIP_ARCHITECTURES} ${BENCH_HIP_OPTIONS_STR}")

add_executable(rocm_bench src/rocm_bench.hip)
set_target_properties(rocm_bench PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON
                                            # rocprofiler_configure must be visible to rocprofiler-register
                                            ENABLE_EXPORTS ON
                                            BUILD_RPATH "${RPP_ROOT}/lib;${ROCPROFILER_SDK_ROOT}/lib")
target_include_directories(rocm_bench PRIVATE ${FKL_DIR}/include ${RPP_INCLUDE_DIR} ${ROCPROFILER_SDK_INCLUDE_DIR})
target_compile_options(rocm_bench PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${BENCH_HIP_OPTIONS}>)
target_compile_definitions(rocm_bench PRIVATE
    BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_FKL_BRANCH="${FKL_BRANCH}" BENCH_HIP_FLAGS="${BENCH_HIP_FLAGS}"
    BENCH_ROCM_PATH="${ROCM_PATH}" BENCH_RPP_LIB="${RPP_LIBRARY}" BENCH_ROCPROFILER_LIB="${ROCPROFILER_SDK_LIBRARY}")
target_link_libraries(rocm_bench PRIVATE ${RPP_LIBRARY} ${ROCPROFILER_SDK_LIBRARY})
