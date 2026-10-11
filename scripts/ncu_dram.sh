#!/usr/bin/env bash
# DRAM traffic of every Figure 1 pipeline with Nsight Compute. Needs root because the NVIDIA driver restricts the GPU
# performance counters to admin users by default (RmProfilingAdminOnly=1):
#
#     sudo scripts/ncu_dram.sh
#
# Paths default to the layout described in README.md (the build and build-nppdx folders of this repository, and
# results/ next to it) and can be overridden with the environment variables NCU, PAPER_BENCH, NPPDX_BENCH and OUT, e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_dram.sh
#
# For every pipeline and N in {1, 16, 100}, profiles one execution (the NVTX range "measure" of the --ncu-target mode
# of the benchmarks) twice: with the caches flushed before every kernel (--cache-control all, cold) and without
# flushing (--cache-control none, warm, like the latency benchmarks). Reports go to results/ncu and are given back to
# the user who ran sudo. Nothing on the system is changed. Summarize afterwards, without sudo, with ncu_summarize.py.
set -u

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this script with sudo: sudo $0" >&2
    exit 1
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(dirname "$REPO")
NCU=${NCU:-/usr/local/cuda/bin/ncu}
PAPER_BENCH=${PAPER_BENCH:-$REPO/build/bvf_bench}
NPPDX_BENCH=${NPPDX_BENCH:-$REPO/build-nppdx/benchmark_figure1_bvf}
OUT=${OUT:-$ROOT/results/ncu}
OWNER=${SUDO_USER:-$(id -un)}
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

mkdir -p "$OUT"
LOG=$OUT/ncu_dram.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <N> <cache control>
    local name=$1 bin=$2 impl=$3 n=$4 cache=$5
    total=$((total + 1))
    echo "[$total] $name" | tee -a "$LOG"
    if ! "$NCU" --nvtx --nvtx-include "measure/" --metrics "$METRICS" --cache-control "$cache" \
            --force-overwrite --export "$OUT/$name" "$bin" "$OUT/scratch" --ncu-target "$impl" "$n" >> "$LOG" 2>&1; then
        echo "    FAILED, see $LOG" | tee -a "$LOG"
        failures=$((failures + 1))
    fi
}

for cache in all none; do
    for n in 1 16 100; do
        for impl in opencv_cuda npp fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf; do
            profile "${impl}_N${n}_cache-${cache}" "$PAPER_BENCH" "$impl" "$n" "$cache"
        done
        for impl in nppdx fkl_bvf; do
            profile "nppdxrepo_${impl}_N${n}_cache-${cache}" "$NPPDX_BENCH" "$impl" "$n" "$cache"
        done
    done
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
