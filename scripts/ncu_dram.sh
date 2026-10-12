#!/usr/bin/env bash
# DRAM traffic of every Figure 1 pipeline with Nsight Compute. Needs root while the NVIDIA driver restricts the GPU
# performance counters to admin users (RmProfilingAdminOnly=1, the default; without it, run it without sudo):
#
#     sudo scripts/ncu_dram.sh
#
# Paths default to the layout described in README.md (the build-<name> folders of this repository, and results/ next
# to it) and can be overridden with the environment variables of scripts/ncu_common.sh and OUT (default ncu/ of
# BVF_RESULTS), e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_dram.sh
# scripts/run_all.sh runs it with the paths of its build and results folders.
#
# For every pipeline (OpenCV CUDA, NPP, FKL VF+HF, FKL BVF), profiles one execution on the regions of iteration 0 (the
# NVTX range "measure" of the --ncu-target mode of the benchmarks) with the caches flushed before every kernel
# (--cache-control all, cold) at every batch size N, and without flushing (--cache-control none, warm: L2 keeps what
# the executions before it, on the regions of iterations 1 to 3, left there, as in the timed loop) at the batch sizes
# of WARM_BATCHES (default 1 and 100). Every profile is taken REPEATS times (default 2; report names end in -<k> after
# the first), because the DRAM counters are device-wide: a GPU that also drives a display adds other processes' traffic
# to some profiles, and ncu_summarize.py keeps the smallest measurement of every pipeline, N and cache control. Reports go to results/ncu and are given back to
# the user who ran sudo. Nothing on the system is changed. Summarize afterwards, without sudo, with ncu_summarize.py.
set -u

# The paths, the workload check, the batch sizes and the profiled regions, shared with the other profiling script
. "$(dirname "$0")/ncu_common.sh"
OUT=${OUT:-$RESULTS/ncu}
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

mkdir -p "$OUT"
# Only this invocation's reports: ncu_summarize.py merges every report of a pipeline, N and cache control in OUT, so a
# report left by an earlier invocation (another REPEATS, another build) would be summarized with these
rm -f "$OUT"/*.ncu-rep
LOG=$OUT/ncu_dram.log
: > "$LOG"
failures=0
total=0

REPEATS=${REPEATS:-2}
profile() { # <report name> <benchmark binary> <implementation> <N>, with the cache control and repeat of the loop below
    local name=$1_cache-$cache$suffix bin=$2 impl=$3 n=$4
    total=$((total + 1))
    echo "[$total] $name $*" | tee -a "$LOG"
    if ! "$NCU" --nvtx --nvtx-include "measure/" --metrics "$METRICS" --cache-control "$cache" \
            --force-overwrite --export "$OUT/$name" "$bin" "$OUT/scratch" --ncu-target "$impl" "$n" \
            >> "$LOG" 2>&1; then
        echo "    FAILED, see $LOG" | tee -a "$LOG"
        failures=$((failures + 1))
    fi
}

for repeat in $(seq "$REPEATS"); do
    suffix=$([ "$repeat" = 1 ] && echo "" || echo "-$repeat")
    cache=all
    for n in $BATCHES; do
        profile_pipelines "$n"
    done
    cache=none
    for n in $WARM_BATCHES; do
        profile_pipelines "$n"
    done
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
# A partial run fails, so that its reports are not summarized as if complete
if [ "$failures" -ne 0 ]; then
    exit 1
fi
