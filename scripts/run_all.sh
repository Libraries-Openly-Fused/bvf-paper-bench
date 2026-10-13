#!/usr/bin/env bash
# Builds and runs every benchmark of this repository, each into its own results folder: FKL vs OpenCV CUDA and FKL vs
# NPP (main run and the GPU only run with the L2 flushed), FKL's ablation (the same, plus the host overhead, the
# per-kernel GPU times and bvf_types, the type of the fused operation), FKL vs NPPDx for every region size (both
# programs: main run, L2 flushed, and the runner-up NPPDx configuration re-timed against FKL), FKL vs OpenCV on the CPU,
# with COMPILE_TIME=1 the compile-time units, and with PROFILE=1 the memory traffic with Nsight Compute
# (scripts/ncu_dram.sh and ncu_l2.sh, their summaries and the traffic model), which needs the driver to let every user
# read the GPU performance counters (RmProfilingAdminOnly=0); otherwise, since those scripts then need root, their
# commands are printed at the end. Never uses sudo itself.
#
#     OpenCV_DIR=<OpenCV build or install> scripts/run_all.sh [<results folder, default ../results>]
#
# The workload is common/bench_geometry.h: regions of 20x10, 32x16 and 48x24 by turns, new every iteration. The NPPDx
# kernels are compiled for one region size, so FKL vs NPPDx is built and run once per size (positions still change every
# iteration).
#
# Environment variables:
#   STEPS        the benchmarks to build and run, among opencv-cuda, npp, ablation, nppdx and opencv-cpu (default all)
#   NPPDX_SIZES  the region sizes of FKL vs NPPDx (default "20x10 32x16 48x24")
#   COMPILE_TIME 1 to also run scripts/compile_time.sh after the runs (long: 10 rounds of every unit): FKL's ablation,
#                the OpenCV and NPP units, and in the NPPDx builds the headers and FKL BVF (first size) and the NPPDx
#                kernel at N = 100 (every size)
#   PROFILE      1 to run the Nsight Compute scripts (needs the opencv-cuda, npp and ablation steps and the GPU
#                performance counters open to every user); without it, their commands are printed at the end
#   BUILD_ROOT   where the build folders go (default this repository): build-<step>, and build-nppdx-<size>
#   CPU_CXX      C++ compiler of the CPU benchmark (default clang++; the README recommends a recent clang)
#   CMAKE_ARGS   more options for every CMake project (for example -DBENCH_ITERATIONS=1 for a quick check). They
#                come before the options this script sets for each project (BENCH_REGIONS, OpenCV_DIR, and CPU_CXX for
#                the CPU benchmark), which win. Setting or unsetting BENCH_REGIONS in them is an error: this script sets
#                it for every project (mixed, or the NPPDx sizes of NPPDX_SIZES)
# Results: <results>/<benchmark folder>, FKL vs NPPDx in <results>/fkl-vs-nppdx/<size> (the runner-up configurations in
# its runner-up/), logs in <results>/logs. The GPU benchmarks run one at a time; leave the GPU otherwise idle.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RESULTS=$(realpath -m "${1:-$(dirname "$REPO")/results}")
STEPS=${STEPS:-"opencv-cuda npp ablation nppdx opencv-cpu"}
NPPDX_SIZES=${NPPDX_SIZES:-"20x10 32x16 48x24"}
COMPILE_TIME=${COMPILE_TIME:-0}
if grep -q "RmProfilingAdminOnly: 0" /proc/driver/nvidia/params 2>/dev/null; then
    COUNTERS_OPEN=1
else
    COUNTERS_OPEN=0
fi
PROFILE=${PROFILE:-0}
if [ "$PROFILE" = 1 ] && [ "$COUNTERS_OPEN" = 0 ]; then
    echo "PROFILE=1 needs the GPU performance counters open to every user (RmProfilingAdminOnly=0); run the printed" \
         "Nsight Compute commands with sudo instead" >&2
    PROFILE=0
fi
BUILD_ROOT=${BUILD_ROOT:-$REPO}
CPU_CXX=${CPU_CXX:-clang++}
read -r -a EXTRA <<< "${CMAKE_ARGS:-}"
# The region sizes are set per project below and name the results folders. configure passes them after CMAKE_ARGS, so
# they win, and CMAKE_ARGS must not name BENCH_REGIONS either, so that a value given there is not silently replaced:
# neither -D (joined, -DBENCH_REGIONS=..., or split, -D BENCH_REGIONS=..., with or without a type) nor -U (joined or
# split, with a glob pattern that matches BENCH_REGIONS, such as -U '*')
previous=""
for arg in "${EXTRA[@]}"; do
    if [[ $previous == -[DU] ]]; then
        option=$previous value=$arg # the split form
        previous=""
    else
        option=${arg:0:2} value=${arg:2}
        previous=$arg
    fi
    # The name of a -D option ends at its type or value; the pattern of a -U option is a glob, as in CMake
    if [[ ( $option == -D && ${value%%[:=]*} == BENCH_REGIONS ) ||
          ( $option == -U && -n $value && BENCH_REGIONS == $value ) ]]; then
        echo "CMAKE_ARGS: $option $value sets or unsets BENCH_REGIONS, which this script sets for every project" \
             "(mixed, and the NPPDx sizes of NPPDX_SIZES)" >&2
        exit 1
    fi
done
GENERATOR=${GENERATOR:-"Unix Makefiles"}

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
if { has_step opencv-cuda || has_step opencv-cpu; } && [ -z "${OpenCV_DIR:-}" ]; then
    echo "Set OpenCV_DIR to the OpenCV build or install folder (the OpenCV benchmarks need it)" >&2
    exit 1
fi
# Every size has its own build, results and log paths, so a size listed twice would overwrite its first measurements
seen=" "
for size in $NPPDX_SIZES; do
    if [[ ! $size =~ ^(20x10|32x16|48x24)$ ]]; then
        echo "NPPDX_SIZES: $size is not one of the region sizes 20x10, 32x16 and 48x24" >&2
        exit 1
    fi
    if [[ $seen == *" $size "* ]]; then
        echo "NPPDX_SIZES: $size is listed twice; its second run would overwrite the first one's results" >&2
        exit 1
    fi
    seen+="$size "
done

# The source folder of every step
declare -A FOLDER=([opencv-cuda]=fkl-vs-opencv-cuda [npp]=fkl-vs-npp [ablation]=fkl-ablation [nppdx]=fkl-vs-nppdx
                   [opencv-cpu]=fkl-vs-opencv-cpu)

# Configures and builds a project. BENCH_REGIONS is a cached CMake option, so every configure sets it (mixed, or the
# NPPDx size): a build folder reused from an earlier configure with another value must not keep it. The project's
# options come after CMAKE_ARGS: CMake applies -D and -U in order, so they win.
configure() { # <source> <build folder> <options...>
    local source=$1 build=$2
    shift 2
    mkdir -p "$build"
    cmake -G "$GENERATOR" -S "$source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          "${EXTRA[@]}" "$@" > "$build/run_all_configure.log" 2>&1 || { cat "$build/run_all_configure.log" >&2; exit 1; }
    cmake --build "$build" -j "$(nproc)" > "$build/run_all_build.log" 2>&1 ||
        { tail -50 "$build/run_all_build.log" >&2; exit 1; }
}

run() { # <log> <command...>: runs a benchmark, keeping its output
    local log=$1
    shift
    echo "    $*"
    "$@" > "$log" 2>&1 || { tail -30 "$log" >&2; echo "FAILED: $*" >&2; exit 1; }
}

LOGS=$RESULTS/logs
mkdir -p "$LOGS"
echo "== Results in $RESULTS"
for step in opencv-cuda npp ablation opencv-cpu; do
    has_step "$step" || continue
    folder=${FOLDER[$step]}
    build=$BUILD_ROOT/build-$step
    out=$RESULTS/$folder
    echo "  $folder ($build)"
    case "$step" in
        opencv-cuda)
            configure "$REPO/$folder" "$build" -DBENCH_REGIONS=mixed -DOpenCV_DIR="$OpenCV_DIR"
            program=benchmark_opencv_cuda ;;
        npp)
            configure "$REPO/$folder" "$build" -DBENCH_REGIONS=mixed
            program=benchmark_npp ;;
        ablation)
            configure "$REPO/$folder" "$build" -DBENCH_REGIONS=mixed
            program=benchmark_fkl_ablation ;;
        opencv-cpu)
            configure "$REPO/$folder" "$build" -DBENCH_REGIONS=mixed -DCMAKE_CXX_COMPILER="$CPU_CXX" \
                      -DOpenCV_DIR="$OpenCV_DIR"
            program=benchmark_opencv_cpu ;;
    esac
    run "$LOGS/$program.log" "$build/$program" "$out"
    if [ "$step" != opencv-cpu ]; then
        run "$LOGS/${program}_l2_flushed.log" "$build/$program" "$out" --gpu-only-l2-flushed
    fi
    if [ "$step" = ablation ]; then
        run "$LOGS/${program}_host_overhead.log" "$build/$program" "$out" --host-overhead
        run "$LOGS/${program}_profile_kernels.log" "$build/$program" "$out" --profile-kernels
        run "$LOGS/bvf_types.log" "$build/bvf_types"
    fi
done
if has_step nppdx; then
    for size in $NPPDX_SIZES; do
        build=$BUILD_ROOT/build-nppdx-$size
        out=$RESULTS/fkl-vs-nppdx/$size
        echo "  fkl-vs-nppdx, $size regions ($build)"
        configure "$REPO/fkl-vs-nppdx" "$build" -DBENCH_REGIONS="$size"
        for program in benchmark_nppdx benchmark_nppdx_npp; do
            run "$LOGS/${program}_$size.log" "$build/$program" "$out"
            run "$LOGS/${program}_${size}_l2_flushed.log" "$build/$program" "$out" --gpu-only-l2-flushed
            # The second fastest configuration of the sweep against FKL, in the same interleaved iterations
            run "$LOGS/${program}_${size}_runner_up_configs.log" python3 "$REPO/scripts/nppdx_runner_up.py" \
                "$out/${program}_nppdx_block_sweep_GPU_NVIDIA.csv" "$out/${program}_runner_up_config.csv"
            run "$LOGS/${program}_${size}_runner_up.log" "$build/$program" "$out/runner-up" --configs \
                "$out/${program}_runner_up_config.csv"
        done
    done
fi
if [ "$COMPILE_TIME" = 1 ]; then
    echo "  Compile time"
    for step in ablation opencv-cuda npp; do
        has_step "$step" || continue
        run "$LOGS/compile_time_$step.log" env SET="$step" BUILD="$BUILD_ROOT/build-$step" \
            "$REPO/scripts/compile_time.sh" "$RESULTS/compile_time/${FOLDER[$step]}"
    done
    if has_step nppdx; then
        first=1
        for size in $NPPDX_SIZES; do
            # The headers and FKL BVF do not depend on the region size: compiled in the first size's build only
            only=$([ "$first" = 1 ] && echo "" || echo "NPPDX 100")
            run "$LOGS/compile_time_nppdx_$size.log" env SET=nppdx BUILD="$BUILD_ROOT/build-nppdx-$size" \
                NPPDX_CONFIG="$RESULTS/fkl-vs-nppdx/$size/benchmark_nppdx_nppdx_config_GPU_NVIDIA.csv" ONLY="$only" \
                "$REPO/scripts/compile_time.sh" "$RESULTS/compile_time/fkl-vs-nppdx-$size"
            first=0
        done
    fi
fi
NCU_COMMAND=""
if has_step opencv-cuda && has_step npp && has_step ablation; then
    benches=(OPENCV_CUDA_BENCH="$BUILD_ROOT/build-opencv-cuda/benchmark_opencv_cuda"
             NPP_BENCH="$BUILD_ROOT/build-npp/benchmark_npp"
             ABLATION_BENCH="$BUILD_ROOT/build-ablation/benchmark_fkl_ablation" BVF_RESULTS="$RESULTS")
    if [ "$PROFILE" = 1 ]; then
        echo "  Memory traffic (Nsight Compute)"
        run "$LOGS/ncu_dram_run.log" env "${benches[@]}" "$REPO/scripts/ncu_dram.sh"
        run "$LOGS/ncu_l2_run.log" env "${benches[@]}" "$REPO/scripts/ncu_l2.sh"
        run "$LOGS/ncu_summarize.log" env BVF_RESULTS="$RESULTS" "$REPO/scripts/ncu_summarize.py"
        run "$LOGS/ncu_summarize_l2.log" env BVF_RESULTS="$RESULTS" "$REPO/scripts/ncu_summarize.py" l2
        run "$LOGS/ncu_traffic_model.log" env BVF_RESULTS="$RESULTS" "$REPO/scripts/ncu_traffic_model.py"
    else
        sudo=$([ "$COUNTERS_OPEN" = 1 ] && echo "" || echo "sudo ")
        NCU_COMMAND="${sudo}${benches[*]} $REPO/scripts/ncu_dram.sh
${sudo}${benches[*]} $REPO/scripts/ncu_l2.sh
BVF_RESULTS=$RESULTS $REPO/scripts/ncu_summarize.py && BVF_RESULTS=$RESULTS $REPO/scripts/ncu_summarize.py l2 && \\
     BVF_RESULTS=$RESULTS $REPO/scripts/ncu_traffic_model.py"
    fi
fi

echo "Done: results in $RESULTS"
if [ -n "$NCU_COMMAND" ]; then
    echo "Memory traffic (Nsight Compute; with sudo while RmProfilingAdminOnly=1):"
    echo "$NCU_COMMAND"
fi
