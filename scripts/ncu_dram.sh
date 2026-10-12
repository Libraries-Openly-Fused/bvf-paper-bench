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
# scripts/run_all.sh runs it for every workload, with the paths of its build and results folders.
#
# For every pipeline and every batch size N of the benchmarks, profiles one execution (the NVTX range "measure" of the
# --ncu-target mode of the benchmarks) twice: with the caches flushed before every kernel (--cache-control all, cold) and without
# flushing (--cache-control none, warm, like the latency benchmarks). Reports go to results/ncu and are given back to
# the user who ran sudo. Nothing on the system is changed. Summarize afterwards, without sudo, with ncu_summarize.py.
set -u

# The paths, the workload and configuration checks and the batch sizes, shared with the other profiling script
. "$(dirname "$0")/ncu_common.sh"
OUT=${OUT:-$RESULTS/ncu}
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

mkdir -p "$OUT"
LOG=$OUT/ncu_dram.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <N> [<NPPDx tile> <NPPDx block>], with the cache
            # control of the loop below
    local name=$1_cache-$cache bin=$2 impl=$3 n=$4
    shift 4
    total=$((total + 1))
    echo "[$total] $name $*" | tee -a "$LOG"
    if ! "$NCU" --nvtx --nvtx-include "measure/" --metrics "$METRICS" --cache-control "$cache" \
            --force-overwrite --export "$OUT/$name" "$bin" "$OUT/scratch" --ncu-target "$impl" "$n" "$@" \
            >> "$LOG" 2>&1; then
        echo "    FAILED, see $LOG" | tee -a "$LOG"
        failures=$((failures + 1))
    fi
}

for cache in all none; do
    for n in $BATCHES; do
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
