cmake_minimum_required(VERSION 3.28)
# FKL vs PyTorch (eager, LibTorch's C++ API) on the Figure 1 pipeline, on the CPU and on the GPU (see README.md).
# LibTorch is the one of a PyTorch wheel: give CMake its torch.utils.cmake_prefix_path, for example
#   cmake -S fkl-vs-pytorch -B build-pytorch -DCMAKE_CXX_COMPILER=<clang++> \
#         -DCMAKE_PREFIX_PATH=$(<venv>/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')
# A CUDA wheel needs the CUDA toolkit (nvcc) and builds both programs; a ROCm wheel builds both with FKL's side as HIP;
# a CPU wheel builds the CPU program only.
project(fkl_vs_pytorch LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_cpu.cmake)

# The GPU program, when LibTorch has a GPU backend: AUTO builds it then, OFF never
set(BENCH_PYTORCH_GPU AUTO CACHE STRING "Build the GPU program: AUTO (when LibTorch has CUDA or ROCm) or OFF")
set_property(CACHE BENCH_PYTORCH_GPU PROPERTY STRINGS AUTO OFF)
# The GPU of FKL's code, as every GPU benchmark (common/bench_gpu.cmake, rocm/CMakeLists.txt): CUDA_ARCH for CUDA,
# HIP_ARCH (one target, FKL's default HIP architecture) for ROCm
set(CUDA_ARCH "native" CACHE STRING "CUDA architectures of FKL's code, passed verbatim to CUDA_ARCHITECTURES")
set(HIP_ARCH "gfx1200" CACHE STRING "HIP architecture of FKL's code (one target)")

# LibTorch's CMake folder (<wheel>/share/cmake/Torch): Torch_DIR if given (scripts/run_all.sh gives it at every
# configure, so that a build folder follows TORCH_PYTHON), otherwise the one in CMAKE_PREFIX_PATH. find_package(Torch)
# uses the same folder.
if(NOT Torch_DIR)
    find_path(torchConfigDir TorchConfig.cmake PATH_SUFFIXES Torch NO_CACHE REQUIRED)
    set(Torch_DIR "${torchConfigDir}" CACHE PATH "LibTorch's CMake folder (<wheel>/share/cmake/Torch)")
endif()
# The wheel's full version (with its CUDA or ROCm variant) and git commit, from its version.py, next to those CMake files
file(STRINGS "${Torch_DIR}/../../../version.py" TORCH_VERSION_LINES REGEX "^(__version__|git_version|cuda|hip)[: ]")
set(BENCH_TORCH_VERSION "")
set(BENCH_TORCH_CUDA FALSE)
foreach(line IN LISTS TORCH_VERSION_LINES)
    string(REGEX REPLACE "^([a-z_]+)[^=]*= *'?([^']*)'?$" "\\1 \\2" line "${line}")
    if(line MATCHES "^cuda [0-9]")
        set(BENCH_TORCH_CUDA TRUE)
    endif()
    if(NOT line MATCHES " None$")
        string(APPEND BENCH_TORCH_VERSION "${line}, ")
    endif()
endforeach()
string(REGEX REPLACE ", $" "" BENCH_TORCH_VERSION "${BENCH_TORCH_VERSION}")
message(STATUS "PyTorch: ${BENCH_TORCH_VERSION}")

# LibTorch's CMake for CUDA enables CUDA with the C++ compiler as nvcc's host compiler when that is clang (which nvcc
# may not support: the CPU program wants the latest clang) and adds its own nvcc flags to CMAKE_CUDA_FLAGS (a -gencode
# per GPU of the machine, --expt-* options, diagnostics). CUDA is enabled here first, with nvcc's default host compiler
# or CMAKE_CUDA_HOST_COMPILER, and both are restored after find_package: FKL's code keeps the nvcc options of every
# other GPU benchmark.
if(BENCH_TORCH_CUDA)
    enable_language(CUDA)
endif()
set(BENCH_SAVED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(BENCH_SAVED_CUDA_HOST_COMPILER "${CMAKE_CUDA_HOST_COMPILER}")
find_package(Torch REQUIRED)
set(CMAKE_CUDA_FLAGS "${BENCH_SAVED_CUDA_FLAGS}")
set(CMAKE_CUDA_HOST_COMPILER "${BENCH_SAVED_CUDA_HOST_COMPILER}")

# --- CPU program: PyTorch's CPU kernels against FKL's CPU backends ---
add_cpu_bench_target(benchmark_pytorch_cpu src/benchmark_pytorch_cpu.cpp LIBRARIES ${TORCH_LIBRARIES})
target_include_directories(benchmark_pytorch_cpu_code PRIVATE src)
target_compile_definitions(benchmark_pytorch_cpu_code PRIVATE BENCH_TORCH_VERSION="${BENCH_TORCH_VERSION}")

# --- GPU program: PyTorch on the GPU against FKL BVF ---
set(BENCH_GPU_LANGUAGE "")
if(NOT BENCH_PYTORCH_GPU STREQUAL "OFF")
    if(TARGET torch_hip OR CMAKE_HIP_COMPILER_LOADED)
        set(BENCH_GPU_LANGUAGE HIP)
    elseif(CMAKE_CUDA_COMPILER_LOADED)
        set(BENCH_GPU_LANGUAGE CUDA)
    endif()
endif()
if(BENCH_GPU_LANGUAGE)
    execute_process(COMMAND git -C ${FKL_DIR} rev-parse --abbrev-ref HEAD OUTPUT_VARIABLE FKL_BRANCH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    # FKL's side, in its own library so that it gets none of LibTorch's compile options: the nvcc options of
    # common/bench_gpu.cmake, or the HIP options of rocm/CMakeLists.txt
    add_library(fkl_gpu STATIC src/fkl_gpu.cu)
    set_target_properties(fkl_gpu PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_include_directories(fkl_gpu PRIVATE ${FKL_DIR}/include ${BENCH_COMMON_DIR} src)
    if(BENCH_GPU_LANGUAGE STREQUAL "CUDA")
        set(BENCH_GPU_OPTIONS --threads 0 -rdc=false -split-compile=0 -ftemplate-depth=1000 -lineinfo)
        if(WIN32)
            list(APPEND BENCH_GPU_OPTIONS -Xcompiler=/bigobj -Xcompiler=/Zc:preprocessor)
        endif()
        # One CUDA runtime in the process: the shared one LibTorch links
        set_target_properties(fkl_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                                 CUDA_ARCHITECTURES "${CUDA_ARCH}" CUDA_RUNTIME_LIBRARY Shared)
        set(BENCH_GPU_COMPILER "nvcc ${CMAKE_CUDA_COMPILER_VERSION}")
        # native: the architectures CMake found, as common/bench_gpu.cmake records them
        if(CUDA_ARCH STREQUAL "native")
            string(REPLACE ";" " " nativeList "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
            set(BENCH_GPU_ARCH "CUDA_ARCHITECTURES native (${nativeList})")
        else()
            set(BENCH_GPU_ARCH "CUDA_ARCHITECTURES ${CUDA_ARCH}")
        endif()
        set(BENCH_GPU_LANGUAGE_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELEASE}")
    else()
        # LibTorch's CMake for ROCm has enabled HIP with its compiler and set CMAKE_HIP_ARCHITECTURES to its
        # PYTORCH_ROCM_ARCH: the architecture is set on the target instead
        enable_language(HIP)
        set_source_files_properties(src/fkl_gpu.cu PROPERTIES LANGUAGE HIP)
        set(BENCH_GPU_OPTIONS -ftemplate-depth=1000 -gline-tables-only)
        set_target_properties(fkl_gpu PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON
                                                 HIP_ARCHITECTURES "${HIP_ARCH}")
        set(BENCH_GPU_COMPILER "${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}")
        set(BENCH_GPU_ARCH "--offload-arch=${HIP_ARCH}")
        set(BENCH_GPU_LANGUAGE_FLAGS "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_RELEASE}")
    endif()
    target_compile_options(fkl_gpu PRIVATE $<$<COMPILE_LANGUAGE:${BENCH_GPU_LANGUAGE}>:${BENCH_GPU_OPTIONS}>)
    string(JOIN " " BENCH_GPU_OPTIONS_STR ${BENCH_GPU_OPTIONS})
    # The C++20 option CMake passes: with the language's extensions (its default, unless CMAKE_<LANG>_EXTENSIONS is
    # set) the GNU dialect where the compiler has one (clang for HIP: -std=gnu++20; nvcc: -std=c++20 either way)
    set(extensions ${CMAKE_${BENCH_GPU_LANGUAGE}_EXTENSIONS_DEFAULT})
    if(DEFINED CMAKE_${BENCH_GPU_LANGUAGE}_EXTENSIONS)
        set(extensions ${CMAKE_${BENCH_GPU_LANGUAGE}_EXTENSIONS})
    endif()
    if(extensions)
        set(BENCH_GPU_DIALECT "${CMAKE_${BENCH_GPU_LANGUAGE}20_EXTENSION_COMPILE_OPTION}")
    else()
        set(BENCH_GPU_DIALECT "${CMAKE_${BENCH_GPU_LANGUAGE}20_STANDARD_COMPILE_OPTION}")
    endif()
    set(BENCH_GPU_FLAGS
        "${BENCH_GPU_LANGUAGE_FLAGS} ${BENCH_GPU_DIALECT} ${BENCH_GPU_OPTIONS_STR}, ${BENCH_GPU_ARCH}")
    # The flags become a string literal: CMake's Windows defaults have double quotes (-Xcompiler=" /GR /EHsc"), and a
    # list of architectures (-DCUDA_ARCH='89;120') has semicolons, which would split the compile definition
    string(REPLACE "\"" "'" BENCH_GPU_FLAGS "${BENCH_GPU_FLAGS}")
    string(REPLACE ";" " " BENCH_GPU_FLAGS "${BENCH_GPU_FLAGS}")
    target_compile_definitions(fkl_gpu PRIVATE ${BENCH_GEOMETRY_DEFINITIONS}
        BENCH_FKL_COMMIT="${FKL_COMMIT}" BENCH_FKL_BRANCH="${FKL_BRANCH}"
        BENCH_GPU_COMPILER="${BENCH_GPU_COMPILER}" BENCH_GPU_FLAGS="${BENCH_GPU_FLAGS}")

    # PyTorch's side and the harness, compiled with the C++ compiler against LibTorch
    add_executable(benchmark_pytorch_gpu src/benchmark_pytorch_gpu.cpp)
    set_target_properties(benchmark_pytorch_gpu PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_include_directories(benchmark_pytorch_gpu PRIVATE ${BENCH_COMMON_DIR} src)
    target_compile_definitions(benchmark_pytorch_gpu PRIVATE ${BENCH_GEOMETRY_DEFINITIONS}
        BENCH_ITERATIONS=${BENCH_ITERATIONS} BENCH_WARMUP_ITERATIONS=${BENCH_WARMUP_ITERATIONS}
        BENCH_REPO_COMMIT="${BENCH_REPO_COMMIT}" BENCH_TORCH_VERSION="${BENCH_TORCH_VERSION}")
    target_link_libraries(benchmark_pytorch_gpu PRIVATE fkl_gpu ${TORCH_LIBRARIES})
    message(STATUS "GPU program: FKL's code as ${BENCH_GPU_LANGUAGE} (${BENCH_GPU_COMPILER}, ${BENCH_GPU_ARCH})")
else()
    message(STATUS "GPU program not built: LibTorch has no GPU backend here, or BENCH_PYTORCH_GPU is OFF")
    # A GPU program built by an earlier configuration of this build folder would be left behind, stale: removed, so
    # that the program exists exactly when this configuration builds it
    file(GLOB staleGpuPrograms "${CMAKE_CURRENT_BINARY_DIR}/benchmark_pytorch_gpu${CMAKE_EXECUTABLE_SUFFIX}"
                               "${CMAKE_CURRENT_BINARY_DIR}/*/benchmark_pytorch_gpu${CMAKE_EXECUTABLE_SUFFIX}")
    if(staleGpuPrograms)
        file(REMOVE ${staleGpuPrograms})
    endif()
endif()
