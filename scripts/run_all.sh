#!/usr/bin/env bash
# Builds and runs every benchmark of this repository for every workload of the paper, each into its own results
# folder: the GPU benchmark (all modes but the Nsight Compute targets), the two FKL vs NPPDx benchmarks (with the L2
# flushed too), the CPU benchmark and, with COMPILE_TIME=1, the compile-time units. Without sudo; the Nsight Compute
# scripts, which need root while the driver restricts the GPU performance counters, are printed at the end.
#
#     OpenCV_DIR=<OpenCV build or install> scripts/run_all.sh [<results folder, default ../results>]
#
# Environment variables:
#   GEOMETRIES   the workloads, as <crop W>x<crop H>:<resize W>x<resize H> (default "32x16:64x32 20x10:64x32": the
#                paper's crops resized 1:2 and its earlier 20x10 crops resized 5:16); the results of one go to
#                <results folder>/<crop W>x<crop H>
#   STEPS        what to build and run, among gpu, nppdx, cpu (default "gpu nppdx cpu")
#   COMPILE_TIME 1 to also run compile_time/run.sh for both sets after the benchmarks (long: 10 rounds of every unit)
#   BUILD_ROOT   where the build folders go (default this repository): build-<crop>, build-nppdx-<crop> and
#                build-cpu-<crop>, for example build-32x16
#   CPU_CXX      C++ compiler of the CPU benchmark (default clang++; the README recommends a recent clang)
#   CMAKE_ARGS   more options for every CMake project (for example -DBENCH_ITERATIONS=1 for a quick check)
# The GPU benchmarks run one at a time; leave the GPU otherwise idle while they run.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RESULTS=$(realpath -m "${1:-$(dirname "$REPO")/results}")
GEOMETRIES=${GEOMETRIES:-"32x16:64x32 20x10:64x32"}
STEPS=${STEPS:-"gpu nppdx cpu"}
COMPILE_TIME=${COMPILE_TIME:-0}
BUILD_ROOT=${BUILD_ROOT:-$REPO}
CPU_CXX=${CPU_CXX:-clang++}
read -r -a EXTRA <<< "${CMAKE_ARGS:-}"
GENERATOR=${GENERATOR:-"Unix Makefiles"}

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
if { has_step gpu || has_step cpu; } && [ -z "${OpenCV_DIR:-}" ]; then
    echo "Set OpenCV_DIR to the OpenCV build or install folder (the GPU and CPU benchmarks need it)" >&2
    exit 1
fi

configure() { # <source> <build folder> <geometry options...>: configures and builds a project
    local source=$1 build=$2
    shift 2
    cmake -G "$GENERATOR" -S "$source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          "$@" "${EXTRA[@]}" > "$build.configure.log" 2>&1 || { cat "$build.configure.log" >&2; exit 1; }
    cmake --build "$build" -j "$(nproc)" > "$build.build.log" 2>&1 || { tail -50 "$build.build.log" >&2; exit 1; }
}

run() { # <log> <command...>: runs a benchmark, keeping its output
    local log=$1
    shift
    echo "    $*"
    "$@" > "$log" 2>&1 || { tail -30 "$log" >&2; echo "FAILED: $*" >&2; exit 1; }
}

NCU_COMMANDS=()
for geometry in $GEOMETRIES; do
    crop=${geometry%%:*}
    resize=${geometry##*:}
    options=(-DBENCH_CROP_W="${crop%%x*}" -DBENCH_CROP_H="${crop##*x}" -DBENCH_RESIZE_W="${resize%%x*}"
             -DBENCH_RESIZE_H="${resize##*x}")
    out=$RESULTS/$crop
    gpu=$BUILD_ROOT/build-$crop
    nppdx=$BUILD_ROOT/build-nppdx-$crop
    cpu=$BUILD_ROOT/build-cpu-$crop
    mkdir -p "$out/logs"
    echo "== $crop crops resized to $resize -> $out"
    if has_step gpu; then
        echo "  GPU benchmark ($gpu)"
        mkdir -p "$gpu"
        configure "$REPO" "$gpu" "${options[@]}" -DOpenCV_DIR="$OpenCV_DIR"
        run "$out/logs/bvf_bench.log" "$gpu/bvf_bench" "$out"
        run "$out/logs/bvf_bench_l2_flushed.log" "$gpu/bvf_bench" "$out" --gpu-only-l2-flushed
        run "$out/logs/bvf_bench_profile_kernels.log" "$gpu/bvf_bench" "$out" --profile-kernels
        run "$out/logs/bvf_bench_host_overhead.log" "$gpu/bvf_bench" "$out" --host-overhead
    fi
    if has_step nppdx; then
        echo "  FKL vs NPPDx ($nppdx)"
        mkdir -p "$nppdx"
        configure "$REPO/fkl-vs-nppdx" "$nppdx" "${options[@]}"
        for pair in benchmark_figure1_bvf:nppdx benchmark_figure1_nppdx_npp:nppdx_npp; do
            name=${pair%%:*}
            impl=${pair##*:}
            run "$out/logs/$name.log" "$nppdx/$name" "$out/$impl"
            run "$out/logs/${name}_l2_flushed.log" "$nppdx/$name" "$out/$impl" --gpu-only-l2-flushed
            run "$out/logs/${name}_analyze.log" "$nppdx/$name" "$out/$impl" --analyze
        done
    fi
    if has_step cpu; then
        echo "  CPU benchmark ($cpu)"
        mkdir -p "$cpu"
        configure "$REPO/cpu" "$cpu" "${options[@]}" -DCMAKE_CXX_COMPILER="$CPU_CXX" -DOpenCV_DIR="$OpenCV_DIR"
        run "$out/logs/cpu_bench.log" "$cpu/cpu_bench" "$out/cpu"
    fi
    if [ "$COMPILE_TIME" = 1 ]; then
        echo "  Compile time"
        has_step gpu && run "$out/logs/compile_time.log" env SET=paper BUILD="$gpu" \
            "$REPO/compile_time/run.sh" "$out/compile_time"
        has_step nppdx && run "$out/logs/compile_time_nppdx.log" env SET=nppdx BUILD="$nppdx" BVF_RESULTS="$out" \
            "$REPO/compile_time/run.sh" "$out/compile_time_nppdx"
    fi
    if has_step gpu && has_step nppdx; then
        NCU_COMMANDS+=("sudo PAPER_BENCH=$gpu/bvf_bench NPPDX_BENCH=$nppdx/benchmark_figure1_bvf \\
     NPPDX_NPP_BENCH=$nppdx/benchmark_figure1_nppdx_npp BVF_RESULTS=$out $REPO/scripts/ncu_dram.sh
sudo PAPER_BENCH=$gpu/bvf_bench NPPDX_BENCH=$nppdx/benchmark_figure1_bvf \\
     NPPDX_NPP_BENCH=$nppdx/benchmark_figure1_nppdx_npp BVF_RESULTS=$out $REPO/scripts/ncu_l2.sh
BVF_RESULTS=$out $REPO/scripts/ncu_summarize.py && BVF_RESULTS=$out $REPO/scripts/ncu_summarize.py l2 && \\
     BVF_RESULTS=$out $REPO/scripts/ncu_traffic_model.py")
    fi
done

echo "Done: results in $RESULTS"
if [ ${#NCU_COMMANDS[@]} -gt 0 ]; then
    echo "Memory traffic (Nsight Compute; root while RmProfilingAdminOnly=1), for every workload:"
    printf '%s\n\n' "${NCU_COMMANDS[@]}"
fi
