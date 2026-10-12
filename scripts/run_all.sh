#!/usr/bin/env bash
# Builds and runs every benchmark of this repository for every workload of the paper, each into its own results
# folder: FKL vs OpenCV CUDA, FKL vs NPP and FKL's ablation (all modes but the Nsight Compute targets), FKL vs NPPDx
# (both programs, with the L2 flushed too), FKL vs OpenCV on the CPU, with COMPILE_TIME=1 the compile-time units, and
# with PROFILE=1 the memory traffic with Nsight Compute (scripts/ncu_dram.sh and ncu_l2.sh, their summaries and the
# traffic model), which needs the driver to let every user read the GPU performance counters (RmProfilingAdminOnly=0);
# otherwise, since those scripts then need root, their commands are printed at the end. Never uses sudo itself.
#
#     OpenCV_DIR=<OpenCV build or install> scripts/run_all.sh [<results folder, default ../results>]
#
# Environment variables:
#   GEOMETRIES   the workloads, as <crop W>x<crop H>:<resize W>x<resize H> (default "32x16:64x32 20x10:64x32": the
#                paper's crops resized 1:2 and its earlier 20x10 crops resized 5:16); the results of one go to
#                <results folder>/<crop W>x<crop H>/<benchmark folder>
#   STEPS        the benchmarks to build and run, among opencv-cuda, npp, ablation, nppdx and opencv-cpu (default all)
#   COMPILE_TIME 1 to also run scripts/compile_time.sh for the GPU benchmarks after the runs (long: 10 rounds of every
#                unit)
#   PROFILE      1 to run the Nsight Compute scripts (needs the four GPU steps and the GPU performance counters open to
#                every user); without it, their commands are printed at the end
#   BUILD_ROOT   where the build folders go (default this repository): build-<step>-<crop W>x<crop H>, for example
#                build-npp-32x16
#   CPU_CXX      C++ compiler of the CPU benchmark (default clang++; the README recommends a recent clang)
#   CMAKE_ARGS   more options for every CMake project (for example -DBENCH_ITERATIONS=1 for a quick check)
# The GPU benchmarks run one at a time; leave the GPU otherwise idle while they run.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RESULTS=$(realpath -m "${1:-$(dirname "$REPO")/results}")
GEOMETRIES=${GEOMETRIES:-"32x16:64x32 20x10:64x32"}
STEPS=${STEPS:-"opencv-cuda npp ablation nppdx opencv-cpu"}
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
GENERATOR=${GENERATOR:-"Unix Makefiles"}

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
if { has_step opencv-cuda || has_step opencv-cpu; } && [ -z "${OpenCV_DIR:-}" ]; then
    echo "Set OpenCV_DIR to the OpenCV build or install folder (the OpenCV benchmarks need it)" >&2
    exit 1
fi

# The source folder of every step
declare -A FOLDER=([opencv-cuda]=fkl-vs-opencv-cuda [npp]=fkl-vs-npp [ablation]=fkl-ablation [nppdx]=fkl-vs-nppdx
                   [opencv-cpu]=fkl-vs-opencv-cpu)

configure() { # <source> <build folder> <options...>: configures and builds a project
    local source=$1 build=$2
    shift 2
    mkdir -p "$build"
    cmake -G "$GENERATOR" -S "$source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          "$@" "${EXTRA[@]}" > "$build/run_all_configure.log" 2>&1 || { cat "$build/run_all_configure.log" >&2; exit 1; }
    cmake --build "$build" -j "$(nproc)" > "$build/run_all_build.log" 2>&1 ||
        { tail -50 "$build/run_all_build.log" >&2; exit 1; }
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
    mkdir -p "$out/logs"
    echo "== $crop crops resized to $resize -> $out"
    for step in opencv-cuda npp ablation nppdx opencv-cpu; do
        has_step "$step" || continue
        folder=${FOLDER[$step]}
        build=$BUILD_ROOT/build-$step-$crop
        echo "  $folder ($build)"
        case "$step" in
            opencv-cuda)
                configure "$REPO/$folder" "$build" "${options[@]}" -DOpenCV_DIR="$OpenCV_DIR"
                programs=(benchmark_opencv_cuda) ;;
            npp)
                configure "$REPO/$folder" "$build" "${options[@]}"
                programs=(benchmark_npp) ;;
            ablation)
                configure "$REPO/$folder" "$build" "${options[@]}"
                programs=(benchmark_fkl_ablation)
                run "$out/logs/bvf_types.log" "$build/bvf_types" ;;
            nppdx)
                configure "$REPO/$folder" "$build" "${options[@]}"
                programs=(benchmark_nppdx benchmark_nppdx_npp) ;;
            opencv-cpu)
                configure "$REPO/$folder" "$build" "${options[@]}" -DCMAKE_CXX_COMPILER="$CPU_CXX" \
                          -DOpenCV_DIR="$OpenCV_DIR"
                programs=(benchmark_opencv_cpu) ;;
        esac
        for program in "${programs[@]}"; do
            run "$out/logs/$program.log" "$build/$program" "$out/$folder"
            case "$step" in
                opencv-cuda|npp|ablation)
                    run "$out/logs/${program}_l2_flushed.log" "$build/$program" "$out/$folder" --gpu-only-l2-flushed
                    run "$out/logs/${program}_profile_kernels.log" "$build/$program" "$out/$folder" --profile-kernels
                    ;;
                nppdx)
                    run "$out/logs/${program}_l2_flushed.log" "$build/$program" "$out/$folder" --gpu-only-l2-flushed
                    run "$out/logs/${program}_analyze.log" "$build/$program" "$out/$folder" --analyze
                    ;;
            esac
        done
        if [ "$step" = ablation ]; then
            run "$out/logs/benchmark_fkl_ablation_host_overhead.log" "$build/benchmark_fkl_ablation" \
                "$out/$folder" --host-overhead
        fi
    done
    if [ "$COMPILE_TIME" = 1 ]; then
        echo "  Compile time"
        for step in ablation opencv-cuda npp nppdx; do
            has_step "$step" || continue
            run "$out/logs/compile_time_$step.log" env SET="$step" BUILD="$BUILD_ROOT/build-$step-$crop" \
                BVF_RESULTS="$out" "$REPO/scripts/compile_time.sh" "$out/compile_time/${FOLDER[$step]}"
        done
    fi
    if has_step opencv-cuda && has_step npp && has_step ablation && has_step nppdx; then
        benches=(OPENCV_CUDA_BENCH="$BUILD_ROOT/build-opencv-cuda-$crop/benchmark_opencv_cuda"
                 NPP_BENCH="$BUILD_ROOT/build-npp-$crop/benchmark_npp"
                 ABLATION_BENCH="$BUILD_ROOT/build-ablation-$crop/benchmark_fkl_ablation"
                 NPPDX_BENCH="$BUILD_ROOT/build-nppdx-$crop/benchmark_nppdx"
                 NPPDX_NPP_BENCH="$BUILD_ROOT/build-nppdx-$crop/benchmark_nppdx_npp" BVF_RESULTS="$out")
        if [ "$PROFILE" = 1 ]; then
            echo "  Memory traffic (Nsight Compute)"
            run "$out/logs/ncu_dram_run.log" env "${benches[@]}" "$REPO/scripts/ncu_dram.sh"
            run "$out/logs/ncu_l2_run.log" env "${benches[@]}" "$REPO/scripts/ncu_l2.sh"
            run "$out/logs/ncu_summarize.log" env BVF_RESULTS="$out" "$REPO/scripts/ncu_summarize.py"
            run "$out/logs/ncu_summarize_l2.log" env BVF_RESULTS="$out" "$REPO/scripts/ncu_summarize.py" l2
            run "$out/logs/ncu_traffic_model.log" env BVF_RESULTS="$out" "$REPO/scripts/ncu_traffic_model.py"
        else
            sudo=$([ "$COUNTERS_OPEN" = 1 ] && echo "" || echo "sudo ")
            NCU_COMMANDS+=("${sudo}${benches[*]} $REPO/scripts/ncu_dram.sh
${sudo}${benches[*]} $REPO/scripts/ncu_l2.sh
BVF_RESULTS=$out $REPO/scripts/ncu_summarize.py && BVF_RESULTS=$out $REPO/scripts/ncu_summarize.py l2 && \\
     BVF_RESULTS=$out $REPO/scripts/ncu_traffic_model.py")
        fi
    fi
done

echo "Done: results in $RESULTS"
if [ ${#NCU_COMMANDS[@]} -gt 0 ]; then
    echo "Memory traffic (Nsight Compute; with sudo while RmProfilingAdminOnly=1), for every workload:"
    printf '%s\n\n' "${NCU_COMMANDS[@]}"
fi
