#!/bin/bash
# All the AMD benchmarks in one command, for the bare-metal run:   rocm/scripts/run_all.sh <results dir>
# Experiment 3 (neighborhood operations, rocm_neighborhood) instead:  EXPERIMENT=3 rocm/scripts/run_all.sh <results dir>
#
# In order: a platform snapshot, the builds, the prerequisite checks (kernel dispatch tracing, hardware counters), the
# GPU benchmark (CPU + GPU pass, then in its profiling child process the kernel counts, GPU time only and GPU time only
# with L2 + MALL flushed), RPP's resize convention, the hardware counter collection (Experiment 2) and the CPU
# benchmark (RPP HOST vs FKL CPU). Every step logs to <results dir>/logs. The workload (region sizes, batch sizes, the
# regions of every iteration) is the one of common/bench_geometry.h; BENCH_CMAKE_ARGS passes another BENCH_REGIONS to
# both builds.
#
# Environment (all optional):
#   ROCM_PATH       ROCm installation (default /opt/rocm/core-10.1)
#   ROCM_EXTRA      prefix with RPP, rocprofiler-sdk and amd-smi when they are not in ROCM_PATH (see env.sh)
#   FKL_GPU_DIR     FusedKernelLibrary for the GPU benchmark, default <repo>/fkl (the submodule, 63782f6)
#   FKL_CPU_DIR     FusedKernelLibrary with its thread-pool submodule for the CPU benchmark, default <repo>/fkl
#   CXX_CPU         C++ compiler of the CPU benchmark (default: the newest clang++-NN found, else clang++)
#   COUNTERS        hardware counters for Experiment 2 (default "FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum":
#                   rocprofiler-sdk 1.4.1 defines FETCH_SIZE for gfx12 but not WRITE_SIZE, which summarize_counters.py
#                   derives from the two GL2C_EA write counters with the GPU's write request size: 256 B on gfx12)
#   BENCH_CMAKE_ARGS extra CMake arguments of both builds, e.g. "-DBENCH_REGIONS=32x16" (default none: mixed sizes)
#   BENCH_MALL_KIB  MALL (Infinity Cache) size in KiB, if rocminfo / amd-smi cannot report it
#   EXPERIMENT      1 (default): Experiments 1 and 2 and the CPU benchmark, as above. 3: Experiment 3 only: the GPU
#                   build, the kernel tracing check and rocm_neighborhood (into <results dir>/neighborhood), with FKL at
#                   the commit recorded for the repository's fkl submodule; no hardware counters, no CPU benchmark
# The hardware counter step needs counters that count: on gfx1200 only at a profiling performance level (at "auto" the
# GL2C counters read 0; sudo amd-smi set -l STABLE_STD, which also pins the shader clock: 1494 MHz on the RX 9060). The
# script checks it before the GPU benchmark with rocprof_counters.sh --check (FETCH_SIZE of a cache flush kernel) and
# stops if they read 0. To time the benchmarks at another level, run with ALLOW_NO_COUNTERS=1 there and collect
# Experiment 2 afterwards at a profiling level with the rocprof_counters.sh command that the script prints at the end.
# SKIP_COUNTERS=1 skips Experiment 2 (the hardware counters) altogether, without checking them: for a run whose results
# do not use AMD memory traffic.
# Development only (the results are then not the paper's): ALLOW_NO_KERNEL_TRACE=1, ALLOW_NO_COUNTERS=1, ALLOW_BUSY=1,
# ALLOW_OTHER_FKL=1, SKIP_CPU=1
set -uo pipefail

# The usage check in this shell: inside $(...) a failed ${1:?} would only end the subshell
: "${1:?usage: rocm/scripts/run_all.sh <results dir>}"
OUT=$(realpath -m "$1")
# A new or empty results directory: files of an earlier run (counters, GPU time only, CPU results that this run might
# not write) must never sit next to this run's
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "run_all.sh: STOP: $OUT is not empty: give a new or empty results directory" >&2
    exit 1
fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
. "$REPO/rocm/scripts/env.sh"
EXPERIMENT=${EXPERIMENT:-1}
[ "$EXPERIMENT" = 1 ] || [ "$EXPERIMENT" = 3 ] || { echo "run_all.sh: STOP: EXPERIMENT is 1 or 3, not $EXPERIMENT" >&2; exit 1; }
# The paper's FKL commit of Experiments 1 and 2 (and of the CPU benchmark); Experiment 3 uses the commit recorded for
# the repository's fkl submodule, which has the batched MorphologyDPP
if [ "$EXPERIMENT" = 1 ]; then
    FKL_PAPER_COMMIT=63782f6
else
    FKL_PAPER_COMMIT=$(git -C "$REPO" ls-tree HEAD fkl | awk '{print substr($3, 1, 7)}')
fi
FKL_GPU_DIR=${FKL_GPU_DIR:-$REPO/fkl}
FKL_CPU_DIR=${FKL_CPU_DIR:-$REPO/fkl}
# Experiment 3 has no CPU benchmark
[ "$EXPERIMENT" = 3 ] && SKIP_CPU=1
if [ -z "${CXX_CPU:-}" ]; then
    CXX_CPU=$(ls /usr/bin/clang++-[0-9]* 2>/dev/null | sort -t- -k2 -n | tail -1)
    CXX_CPU=${CXX_CPU:-clang++}
fi
export COUNTERS=${COUNTERS:-"FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum"}
# RPP's HOST backend uses libomp, whose workers spin 200 ms after every parallel region by default: 100 us, like FKL's
# thread pool, so that the CPU benchmark's idle gap puts them to sleep (rpp_cpu_bench sets the same default)
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-100us}
BUILD_GPU=$REPO/build-rocm
BUILD_CPU=$REPO/build-rocm-cpu
mkdir -p "$OUT/logs"

die() { echo -e "\nrun_all.sh: STOP: $*" >&2; exit 1; }
step() {
    local name=$1; shift
    echo "=== $(date -u +%H:%M:%S) $name"
    local t0=$(date +%s)
    if ! "$@" > "$OUT/logs/$name.log" 2>&1; then
        tail -20 "$OUT/logs/$name.log" >&2
        die "step $name failed (log: $OUT/logs/$name.log)"
    fi
    echo "    done in $(( $(date +%s) - t0 )) s"
}
allowed() { [ "${!1:-0}" = 1 ]; }

# ---------------------------------------------------------------- prerequisites
[ -x "$ROCM_PATH/bin/amdclang++" ] || die "amdclang++ not found in $ROCM_PATH (set ROCM_PATH)"
[ -f "$ROCM_EXTRA/include/rpp/rpp.h" ] && [ -e "$ROCM_EXTRA/lib/librpp.so" ] ||
    die "RPP not found in $ROCM_EXTRA (install amdrocm-rpp10.1 and amdrocm-rpp-dev10.1 or set ROCM_EXTRA, see rocm/scripts/env.sh)"
[ -f "$ROCM_EXTRA/include/rocprofiler-sdk/rocprofiler.h" ] && [ -e "$ROCM_EXTRA/lib/librocprofiler-sdk.so" ] ||
    die "rocprofiler-sdk not found in $ROCM_EXTRA (install amdrocm-profiler-base10.1 or set ROCM_EXTRA)"
command -v amd-smi > /dev/null || die "amd-smi not found (install amdrocm-amdsmi10.1)"
command -v rocminfo > /dev/null || die "rocminfo not found"
[ "$EXPERIMENT" = 3 ] || allowed SKIP_COUNTERS || command -v rocprofv3 > /dev/null ||
    die "rocprofv3 not found (install amdrocm-profiler-base10.1)"
allowed SKIP_CPU || command -v "$CXX_CPU" > /dev/null || die "C++ compiler $CXX_CPU not found (set CXX_CPU)"
FKL_GPU_COMMIT=$(git -C "$FKL_GPU_DIR" rev-parse --short=7 HEAD 2>/dev/null) && [ -f "$FKL_GPU_DIR/CMakeLists.txt" ] ||
    die "no FKL checkout in $FKL_GPU_DIR (git -C $REPO submodule update --init --recursive fkl)"
FKL_DIRS=("$FKL_GPU_DIR")
[ "$FKL_GPU_COMMIT" = "$FKL_PAPER_COMMIT" ] || allowed ALLOW_OTHER_FKL ||
    die "FKL_GPU_DIR is at $FKL_GPU_COMMIT, not $FKL_PAPER_COMMIT (the FKL commit of experiment $EXPERIMENT)"
if ! allowed SKIP_CPU; then
    FKL_CPU_COMMIT=$(git -C "$FKL_CPU_DIR" rev-parse --short=7 HEAD 2>/dev/null) && [ -f "$FKL_CPU_DIR/CMakeLists.txt" ] ||
        die "no FKL checkout in $FKL_CPU_DIR (git -C $REPO submodule update --init --recursive fkl)"
    [ "$FKL_CPU_COMMIT" = "$FKL_PAPER_COMMIT" ] || allowed ALLOW_OTHER_FKL ||
        die "FKL_CPU_DIR is at $FKL_CPU_COMMIT, not $FKL_PAPER_COMMIT"
    FKL_DIRS+=("$FKL_CPU_DIR")
fi
for dir in "${FKL_DIRS[@]}"; do
    # Unmodified: no changes to tracked files, submodules at their recorded commits
    [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] || allowed ALLOW_OTHER_FKL ||
        die "$dir has local modifications (git -C $dir status)"
    # git submodule status: '+' another commit, '-' not initialized, 'U' merge conflicts
    ! git -C "$dir" submodule status 2>/dev/null | grep -q '^[-+U]' || allowed ALLOW_OTHER_FKL ||
        die "$dir: a submodule is not at its recorded commit (git -C $dir submodule status)"
done
allowed SKIP_CPU || [ -n "$(ls -A "$FKL_CPU_DIR/thread-pool/include" 2>/dev/null)" ] ||
    die "$FKL_CPU_DIR/thread-pool is empty (git -C $FKL_CPU_DIR submodule update --init)"

# ---------------------------------------------------------------- platform snapshot, idle machine
{
    echo "run_all.sh started $(date -u '+%Y-%m-%d %H:%M:%S UTC'), experiment $EXPERIMENT"
    echo "--- uname"; uname -a
    echo "--- os"; . /etc/os-release && echo "$PRETTY_NAME"
    echo "--- environment"; grep -qi microsoft /proc/version && echo "WSL2" || echo "bare metal"
    echo "--- uptime / load"; uptime
    echo "--- CPU"; lscpu | sed -n '1,25p'
    echo "--- ROCm"; cat "$ROCM_PATH/.info/version" 2>/dev/null; echo "ROCM_PATH=$ROCM_PATH ROCM_EXTRA=$ROCM_EXTRA"
    echo "--- amd-smi static"; amd-smi static --gpu 0 2>&1
    echo "--- amd-smi metric (clocks, power: performance level and clock lock state, not changed by this script)"
    amd-smi metric --gpu 0 2>&1
    echo "--- amd-smi performance level"; amd-smi metric --gpu 0 --perf-level 2>&1
    echo "--- rocm-smi --showperflevel"; rocm-smi --showperflevel 2>&1 || echo "(rocm-smi not available)"
    echo "--- GPU processes"; amd-smi process 2>&1 || true
    echo "--- rocminfo (GPU agent)"; rocminfo 2>/dev/null | awk '/Device Type: *GPU/{g=1} g' | sed -n '1,80p'
    echo "--- repository"; git -C "$REPO" log -1 --format='%H %s'; git -C "$REPO" status --short
    echo "--- FKL GPU $FKL_GPU_DIR"; git -C "$FKL_GPU_DIR" log -1 --format='%H %d %s'
    if ! allowed SKIP_CPU; then
        echo "--- FKL CPU $FKL_CPU_DIR"; git -C "$FKL_CPU_DIR" log -1 --format='%H %d %s'; git -C "$FKL_CPU_DIR" submodule status
    fi
    echo "--- compilers"; "$ROCM_PATH/bin/amdclang++" --version | head -1
    allowed SKIP_CPU || "$CXX_CPU" --version | head -1
} > "$OUT/platform_prerun.txt" 2>&1
echo "Platform snapshot: $OUT/platform_prerun.txt"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($LOAD >= 0.5)}"; then
    allowed ALLOW_BUSY || die "load average $LOAD >= 0.5: the machine is not idle (close everything else, or ALLOW_BUSY=1)"
    echo "WARNING: load average $LOAD >= 0.5 (ALLOW_BUSY=1)"
fi

# ---------------------------------------------------------------- builds
# Every configure starts from an empty CMake cache (cmake --fresh), so that no value cached in an existing build
# directory by an earlier run or a manual configure (BENCH_REGIONS, the iteration counts, compiler flags, architecture
# flags) leaks into this run: the defaults of the CMake projects, unless BENCH_CMAKE_ARGS sets them
GEOMETRY_ARGS="${BENCH_CMAKE_ARGS:-}"
step build_gpu bash -c "cmake --fresh -S '$REPO/rocm' -B '$BUILD_GPU' -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER='$ROCM_PATH/bin/amdclang++' -DCMAKE_HIP_COMPILER='$ROCM_PATH/bin/amdclang++' \
    -DROCM_PATH='$ROCM_PATH' -DFKL_DIR='$FKL_GPU_DIR' -DRPP_ROOT='$ROCM_EXTRA' -DROCPROFILER_SDK_ROOT='$ROCM_EXTRA' \
    $GEOMETRY_ARGS && cmake --build '$BUILD_GPU'"
if ! allowed SKIP_CPU; then
    step build_cpu bash -c "cmake --fresh -S '$REPO/rocm/cpu' -B '$BUILD_CPU' -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER='$CXX_CPU' -DROCM_PATH='$ROCM_PATH' -DFKL_DIR='$FKL_CPU_DIR' -DRPP_ROOT='$ROCM_EXTRA' \
        $GEOMETRY_ARGS && cmake --build '$BUILD_CPU'"
fi
BENCH=$BUILD_GPU/rocm_bench
[ "$EXPERIMENT" = 3 ] && BENCH=$BUILD_GPU/rocm_neighborhood
[ -x "$BENCH" ] || die "$BENCH was not built (Experiment 3 needs FKL with the batched MorphologyDPP, see $OUT/logs/build_gpu.log)"
WORKLOAD=$("$BENCH" --workload) || die "$BENCH --workload failed"
if ! allowed SKIP_CPU && [ "$WORKLOAD" != "$("$BUILD_CPU/rpp_cpu_bench" --workload)" ]; then
    die "rocm_bench and rpp_cpu_bench have different workloads (region sizes or batch sizes): build both from the same checkout and options"
fi
echo "Workload: $WORKLOAD"

# ---------------------------------------------------------------- what the profiler supports here
TRACING=1
if ! BENCH_ROCPROFILER=1 "$BENCH" "$OUT/logs/tmp" --check-kernel-tracing > "$OUT/logs/check_kernel_tracing.log" 2>&1; then
    TRACING=0
    allowed ALLOW_NO_KERNEL_TRACE ||
        die "rocprofiler-sdk delivers no kernel dispatch records here: GPU time only cannot be measured (WSL2? see $OUT/logs/check_kernel_tracing.log; ALLOW_NO_KERNEL_TRACE=1 to continue without)"
    echo "WARNING: no kernel dispatch tracing: GPU time only will not be measured (ALLOW_NO_KERNEL_TRACE=1)"
fi
COUNTERS_OK=1
if [ "$EXPERIMENT" = 3 ]; then
    # Neither rocprofv3 nor any counter query runs
    COUNTERS_OK=0
    echo "Experiment 3: no hardware counters"
elif allowed SKIP_COUNTERS; then
    # Neither rocprofv3 nor any counter query runs
    COUNTERS_OK=0
    echo "Experiment 2 (hardware counters) skipped (SKIP_COUNTERS=1)"
else
    rocprofv3 -L > "$OUT/logs/rocprofv3_available_counters.txt" 2>&1 || true
    for c in $COUNTERS; do
        grep -qw "$c" "$OUT/logs/rocprofv3_available_counters.txt" || COUNTERS_OK=0
    done
    if [ $COUNTERS_OK = 0 ]; then
        allowed ALLOW_NO_COUNTERS ||
            die "rocprofv3 does not list the counters '$COUNTERS' for this GPU (list: $OUT/logs/rocprofv3_available_counters.txt): set COUNTERS to memory read/write traffic counters it lists, or ALLOW_NO_COUNTERS=1 to skip Experiment 2"
        echo "WARNING: counters '$COUNTERS' not available: Experiment 2 skipped (ALLOW_NO_COUNTERS=1)"
    fi
    if [ $COUNTERS_OK = 1 ]; then
        # Do the counters count at this performance level? (gfx1200: not at "auto", which leaves clock gating on)
        CHECK_STATUS=0
        "$REPO/rocm/scripts/rocprof_counters.sh" --check "$OUT/logs/counter_check" "$BENCH" \
            > "$OUT/logs/counter_check.log" 2>&1 || CHECK_STATUS=$?
        if [ $CHECK_STATUS != 0 ]; then
            COUNTERS_OK=0
            PERF_LEVEL=$(amd-smi metric --gpu 0 --perf-level 2>/dev/null | sed -n 's/^ *PERF_LEVEL: *//p' | head -1)
            if [ $CHECK_STATUS = 3 ]; then
                REASON="the hardware counters read 0 at performance level ${PERF_LEVEL:-unknown} (FETCH_SIZE of a cache flush kernel, see $OUT/logs/counter_check.log): set a profiling level, e.g. sudo amd-smi set -l STABLE_STD"
            else
                REASON="the hardware counter check failed (exit status $CHECK_STATUS, see $OUT/logs/counter_check.log)"
            fi
            allowed ALLOW_NO_COUNTERS || die "$REASON, or ALLOW_NO_COUNTERS=1 to skip Experiment 2"
            echo "WARNING: $REASON: Experiment 2 skipped (ALLOW_NO_COUNTERS=1)"
        else
            tail -1 "$OUT/logs/counter_check.log"
        fi
    fi
fi

# ---------------------------------------------------------------- GPU
if [ "$EXPERIMENT" = 3 ]; then
    step gpu_neighborhood "$BENCH" "$OUT/neighborhood"
else
    step gpu_resize_convention "$BENCH" "$OUT/gpu" --resize-convention
    step gpu_main "$BENCH" "$OUT/gpu"
fi
if [ $COUNTERS_OK = 1 ]; then
    # The counters of the RPP pipeline that the GPU benchmark's selection pass chose
    BENCH_RPP_VARIANT=$(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
                                 $col["chosen"] == "yes" { print $col["variant"]; exit }' "$OUT/gpu/rpp_selection.csv")
    [ -n "$BENCH_RPP_VARIANT" ] || die "no chosen RPP variant in $OUT/gpu/rpp_selection.csv"
    export BENCH_RPP_VARIANT
    step gpu_counters "$REPO/rocm/scripts/rocprof_counters.sh" "$OUT/traffic" "$BENCH"
fi

# ---------------------------------------------------------------- CPU
if ! allowed SKIP_CPU; then
    step cpu_resize_convention "$BUILD_CPU/rpp_cpu_bench" "$OUT/cpu" --resize-convention
    step cpu_main "$BUILD_CPU/rpp_cpu_bench" "$OUT/cpu"
fi

{ echo "run_all.sh finished $(date -u '+%Y-%m-%d %H:%M:%S UTC')"; uptime; amd-smi metric --gpu 0 2>&1; } > "$OUT/platform_postrun.txt"
echo "All done: $OUT (kernel dispatch tracing: $TRACING, counters: $COUNTERS_OK)"
if [ $COUNTERS_OK = 0 ] && ! allowed SKIP_COUNTERS && [ "$EXPERIMENT" = 1 ]; then
    echo "Experiment 2 (hardware counters) was NOT collected. At a performance level where the counters count" \
         "(e.g. sudo amd-smi set -l STABLE_STD):"
    echo "    COUNTERS='$COUNTERS' BENCH_RPP_VARIANT=$(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        $col["chosen"] == "yes" { print $col["variant"]; exit }' "$OUT/gpu/rpp_selection.csv" 2>/dev/null)" \
         "$REPO/rocm/scripts/rocprof_counters.sh $OUT/traffic $BENCH"
fi
