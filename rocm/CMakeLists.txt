cmake_minimum_required(VERSION 3.28)
# Radeon RX 9060 (gfx1200) by default, FKL's default HIP architecture (cmake/libs/hip/archs.cmake); one target, since it
# goes into --offload-arch. Set before project(), where CMake would otherwise detect the GPU of the machine
set(CMAKE_HIP_ARCHITECTURES gfx1200 CACHE STRING "HIP architecture of the GPU benchmark (one target)")
# ROCm port of the GPU benchmarks of common/bench_gpu.h: FKL's HIP backend against RPP, on an AMD GPU.
# Configure with amdclang++ as both the C++ and the HIP compiler, as FKL's ROCm CI does, for example:
#   cmake -S rocm -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
#         -DRPP_ROOT=<prefix with include/rpp and lib/librpp.so> -DROCPROFILER_SDK_ROOT=<prefix with rocprofiler-sdk>
project(bvf_paper_bench_rocm LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The workload of every benchmark (common/bench_geometry.h): the option BENCH_REGIONS (region sizes: mixed, the paper's,
# or one of 20x10, 32x16, 48x24) and BENCH_GEOMETRY_DEFINITIONS
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_geometry.cmake)
# Timed and warm-up iterations, as in every benchmark of the repository (common/bench_gpu.cmake)
set(BENCH_ITERATIONS 1000 CACHE STRING "Timed iterations of every benchmark (at least 1)")
set(BENCH_WARMUP_ITERATIONS 100 CACHE STRING "Warm-up iterations before the timed ones")

set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fkl" CACHE PATH "FusedKernelLibrary checkout (default: the repository's fkl submodule)")
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
# The paper's FKL commits: 63782f6 for Experiments 1 and 2 (rocm_bench), the commit recorded for the repository's fkl
# submodule for Experiment 3 (rocm_neighborhood, which needs MorphologyDPP's batched box blur)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/.. ls-tree HEAD fkl OUTPUT_VARIABLE FKL_SUBMODULE_ENTRY
                OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]" FKL_SUBMODULE_COMMIT
       "${FKL_SUBMODULE_ENTRY}")
if(NOT FKL_COMMIT MATCHES "^63782f6" AND (NOT FKL_SUBMODULE_COMMIT OR NOT FKL_COMMIT MATCHES "^${FKL_SUBMODULE_COMMIT}"))
    message(WARNING "FKL_DIR (${FKL_DIR}) is at ${FKL_COMMIT}, neither 63782f6 (the FKL commit of Experiments 1 and 2) "
                    "nor ${FKL_SUBMODULE_COMMIT} (the repository's fkl submodule, the FKL commit of Experiment 3)")
endif()
execute_process(COMMAND git -C ${FKL_DIR} rev-parse --abbrev-ref HEAD OUTPUT_VARIABLE FKL_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)
# The commit of this repository (its SHA even at a tag), and whether its checkout has changes, as in
# common/bench_gpu.cmake: both show up in platform.txt
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/.. rev-parse --short HEAD OUTPUT_VARIABLE BENCH_REPO_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/.. status --porcelain --untracked-files=no
                OUTPUT_VARIABLE BENCH_REPO_CHANGES OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(BENCH_REPO_CHANGES)
    string(APPEND BENCH_REPO_COMMIT " (with uncommitted changes)")
endif()

# FKL's own HIP target options (cmake/libs/hip/target_generation.cmake, debug.cmake): HIP_STANDARD 20 (required),
# -ftemplate-depth=1000 (TEMPLATE_DEPTH), -gline-tables-only (ENABLE_HIP_LINE_INFO); Release adds -O3 -DNDEBUG
set(BENCH_HIP_OPTIONS -ftemplate-depth=1000 -gline-tables-only)
string(JOIN " " BENCH_HIP_OPTIONS_STR ${BENCH_HIP_OPTIONS})
set(BENCH_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_RELEASE} -std=gnu++20 --offload-arch=${CMAKE_HIP_ARCHITECTURES} ${BENCH_HIP_OPTIONS_STR}")

# The two programs: rocm_bench (Experiments 1 and 2, Figure 1) and rocm_neighborhood (Experiment 3, which includes
# rocm_bench.hip for its harness), with the same options
function(add_bench_program name source)
    add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON
                                             # rocprofiler_configure must be visible to rocprofiler-register
                                             ENABLE_EXPORTS ON
                                             BUILD_RPATH "${RPP_ROOT}/lib;${ROCPROFILER_SDK_ROOT}/lib")
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common ${FKL_DIR}/include
                                               ${RPP_INCLUDE_DIR} ${ROCPROFILER_SDK_INCLUDE_DIR})
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${BENCH_HIP_OPTIONS}>)
    target_compile_definitions(${name} PRIVATE ${BENCH_GEOMETRY_DEFINITIONS}
        BENCH_ITERATIONS=${BENCH_ITERATIONS} BENCH_WARMUP_ITERATIONS=${BENCH_WARMUP_ITERATIONS}
        BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_FKL_BRANCH="${FKL_BRANCH}" BENCH_REPO_COMMIT="${BENCH_REPO_COMMIT}"
        BENCH_HIP_FLAGS="${BENCH_HIP_FLAGS}"
        BENCH_ROCM_PATH="${ROCM_PATH}" BENCH_RPP_LIB="${RPP_LIBRARY}" BENCH_ROCPROFILER_LIB="${ROCPROFILER_SDK_LIBRARY}")
    target_link_libraries(${name} PRIVATE ${RPP_LIBRARY} ${ROCPROFILER_SDK_LIBRARY})
endfunction()

add_bench_program(rocm_bench src/rocm_bench.hip)
# rocm_neighborhood needs MorphologyDPP's box blur (executeBoxBlur, FKL PR #393, batched over the planes of its input),
# which FKL 63782f6 does not have, so the program is only built with an FKL that has it
file(STRINGS "${FKL_DIR}/include/fused_kernel/algorithms/image_processing/morphology.h" FKL_BOX_BLUR
     REGEX "executeBoxBlur")
if(FKL_BOX_BLUR)
    add_bench_program(rocm_neighborhood src/rocm_neighborhood.hip)
else()
    message(STATUS "rocm_neighborhood (Experiment 3) not built: the MorphologyDPP of FKL_DIR (${FKL_COMMIT}) has no "
                   "executeBoxBlur; it needs the commit of the repository's fkl submodule")
endif()
