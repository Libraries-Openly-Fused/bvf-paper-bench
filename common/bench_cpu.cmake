# The common part of the CPU benchmark projects built on common/bench_cpu.h (fkl-vs-opencv-cpu; fkl-vs-pytorch,
# fkl-vs-halide-cpu and fkl-vs-fern-cpu, added by pull requests #18, #16 and #17): FKL with its CPU backends, the CPU
# code generation flags, the iterations, the workload, and add_cpu_bench_target(), which builds one benchmark with them.
# Include it right
# after project(). Meant to be built with the latest stable clang and the best SIMD the CPU has (CPU_ARCH_FLAGS).

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(BENCH_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(FKL_DIR "${BENCH_COMMON_DIR}/../fkl" CACHE PATH "FusedKernelLibrary checkout (default: the fkl submodule)")
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
include(${BENCH_COMMON_DIR}/bench_geometry.cmake)

execute_process(COMMAND git -C ${FKL_DIR} rev-parse --short HEAD OUTPUT_VARIABLE FKL_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${FKL_DIR}/thread-pool rev-parse --short HEAD OUTPUT_VARIABLE THREAD_POOL_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
# The commit of this repository, and whether its checkout has changes, as in common/bench_gpu.cmake
execute_process(COMMAND git -C ${BENCH_COMMON_DIR}/.. rev-parse --short HEAD OUTPUT_VARIABLE BENCH_REPO_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C ${BENCH_COMMON_DIR}/.. status --porcelain --untracked-files=no
                OUTPUT_VARIABLE BENCH_REPO_CHANGES OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(BENCH_REPO_CHANGES)
    string(APPEND BENCH_REPO_COMMIT " (with uncommitted changes)")
endif()
# The CPU_MULTITHREAD backend uses the thread pool of FKL's thread-pool submodule
find_package(Threads REQUIRED)

separate_arguments(CPU_ARCH_FLAGS_LIST NATIVE_COMMAND "${CPU_ARCH_FLAGS}")
if(MSVC)
    # As FKL's tests with cl and clang-cl; cl has no template depth option, clang-cl takes clang's
    set(BENCH_CPU_EXTRA_FLAGS /bigobj /Zc:preprocessor)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND BENCH_CPU_EXTRA_FLAGS /clang:-ftemplate-depth=1000)
    endif()
    set(BENCH_CPU_STD_FLAG /std:c++20)
else()
    set(BENCH_CPU_EXTRA_FLAGS -ftemplate-depth=1000)
    set(BENCH_CPU_STD_FLAG -std=c++20)
endif()
string(JOIN " " BENCH_CPU_EXTRA_FLAGS_STR ${BENCH_CPU_EXTRA_FLAGS})

# The options of the code of a CPU benchmark: C++20, FKL and common/ in the include path, CPU_ARCH_FLAGS, and the
# definitions common/bench_cpu.h reads
function(bench_cpu_options TARGET_NAME)
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_include_directories(${TARGET_NAME} PRIVATE ${FKL_DIR}/include ${FKL_DIR}/thread-pool/include
                                                      ${BENCH_COMMON_DIR})
    target_compile_options(${TARGET_NAME} PRIVATE ${CPU_ARCH_FLAGS_LIST} ${BENCH_CPU_EXTRA_FLAGS})
    # FK_CPU_MULTITHREAD_ENABLED turns on FKL's CPU_MULTITHREAD backend (what FKL::FKL defines when FKL is configured
    # with it)
    target_compile_definitions(${TARGET_NAME} PRIVATE FK_CPU_MULTITHREAD_ENABLED
        BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_THREAD_POOL_COMMIT="${THREAD_POOL_COMMIT}"
        BENCH_REPO_COMMIT="${BENCH_REPO_COMMIT}"
        BENCH_ITERATIONS=${BENCH_ITERATIONS} BENCH_WARMUP_ITERATIONS=${BENCH_WARMUP_ITERATIONS}
        ${BENCH_GEOMETRY_DEFINITIONS}
        BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        BENCH_CXX_FLAGS="${CMAKE_CXX_FLAGS_RELEASE} ${BENCH_CPU_STD_FLAG} ${CPU_ARCH_FLAGS} ${BENCH_CPU_EXTRA_FLAGS_STR}")
endfunction()

# A CPU benchmark executable: add_cpu_bench_target(<name> <sources>... [LIBRARIES <libraries>...]). FKL's side
# (common/bench_fkl_cpu.cpp, FKL's pipelines on the CPU and CPU_MULTITHREAD backends) is the object library <name>_fkl,
# compiled with these options only, and the program's sources are the object library <name>_code, compiled with these
# options and the usage requirements of LIBRARIES (the compared library; add the program's own include folders and
# definitions to <name>_code). The executable links FKL's objects first, then the program's (Makefile and Ninja
# generators put an executable's own sources before the objects of object libraries, so the executable has none):
# FKL's object is the same in every CPU benchmark, and with its page-aligned first function (bench_fkl_cpu.cpp) every
# FKL function has the same address modulo 4096 in all of them.
function(add_cpu_bench_target TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 BENCH "" "" "LIBRARIES")
    add_library(${TARGET_NAME}_fkl OBJECT ${BENCH_COMMON_DIR}/bench_fkl_cpu.cpp)
    bench_cpu_options(${TARGET_NAME}_fkl)
    add_library(${TARGET_NAME}_code OBJECT ${BENCH_UNPARSED_ARGUMENTS})
    bench_cpu_options(${TARGET_NAME}_code)
    target_link_libraries(${TARGET_NAME}_code PRIVATE ${BENCH_LIBRARIES})
    add_executable(${TARGET_NAME} $<TARGET_OBJECTS:${TARGET_NAME}_fkl> $<TARGET_OBJECTS:${TARGET_NAME}_code>)
    set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads ${BENCH_LIBRARIES})
endfunction()
