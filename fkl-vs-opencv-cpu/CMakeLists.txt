cmake_minimum_required(VERSION 3.28)
# FKL vs OpenCV on the CPU: OpenCV's CPU functions vs FKL's CPU backends (BVF), Figure 1 pipeline (see README.md).
# Meant to be built with the latest stable clang and the best SIMD the CPU has, for example:
#   cmake -S fkl-vs-opencv-cpu -B build-opencv-cpu -DCMAKE_CXX_COMPILER=<path to clang++> -DOpenCV_DIR=<OpenCV>
project(fkl_vs_opencv_cpu LANGUAGES CXX)
# FKL, CPU_ARCH_FLAGS, the iterations, the workload and add_cpu_bench_target, shared by every CPU benchmark
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/bench_cpu.cmake)

# OpenCVConfig.cmake still calls find_package(CUDA) when OpenCV was built with CUDA, which needs FindCUDA
cmake_policy(SET CMP0146 OLD)
find_package(OpenCV 4.14 REQUIRED COMPONENTS core imgproc)

add_cpu_bench_target(benchmark_opencv_cpu src/benchmark_opencv_cpu.cpp LIBRARIES opencv_core opencv_imgproc)
