#!/usr/bin/env bash
# Memory traffic between the SMs and the L2 cache of every Figure 1 pipeline with Nsight Compute: every byte the
# kernels read or write, whether it then stays in L2 or goes to DRAM. Needs root while the NVIDIA driver restricts the
# GPU performance counters to admin users (RmProfilingAdminOnly=1, the default; without it, run it without sudo):
#
#     sudo scripts/ncu_l2.sh
#
# Paths default to the layout described in README.md (the build-<name> folders of this repository, and results/ next
# to it) and can be overridden with the environment variables of scripts/ncu_common.sh and OUT (default ncu_l2/ of
# BVF_RESULTS), e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_l2.sh
# scripts/run_all.sh runs it for every workload, with the paths of its build and results folders.
#
# For every pipeline and every batch size N of the benchmarks, profiles one execution (the NVTX range "measure" of the
# --ncu-target mode of the benchmarks) with the caches flushed before every kernel. Reports go to results/ncu_l2 and are given back to
# the user who ran sudo. Nothing on the system is changed. Summarize afterwards, without sudo, with
# ncu_summarize.py l2.
set -u

# The paths, the workload and configuration checks and the batch sizes, shared with the other profiling script
. "$(dirname "$0")/ncu_common.sh"
OUT=${OUT:-$RESULTS/ncu_l2}
METRICS=lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

mkdir -p "$OUT"
LOG=$OUT/ncu_l2.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <N> [<NPPDx tile> <NPPDx block>]
    local name=$1_cache-all bin=$2 impl=$3 n=$4
    shift 4
    total=$((total + 1))
    echo "[$total] $name $*" | tee -a "$LOG"
    if ! "$NCU" --nvtx --nvtx-include "measure/" --metrics "$METRICS" --cache-control all \
            --force-overwrite --export "$OUT/$name" "$bin" "$OUT/scratch" --ncu-target "$impl" "$n" "$@" \
            >> "$LOG" 2>&1; then
        echo "    FAILED, see $LOG" | tee -a "$LOG"
        failures=$((failures + 1))
    fi
}

for n in $BATCHES; do
    profile_pipelines "$n"
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
