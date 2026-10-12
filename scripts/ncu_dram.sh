#!/usr/bin/env bash
# DRAM traffic of every Figure 1 pipeline with Nsight Compute. Needs root because the NVIDIA driver restricts the GPU
# performance counters to admin users by default (RmProfilingAdminOnly=1):
#
#     sudo scripts/ncu_dram.sh
#
# Paths default to the layout described in README.md (the build and build-nppdx folders of this repository, and
# results/ next to it) and can be overridden with the environment variables NCU, PAPER_BENCH, NPPDX_BENCH, NPPDX_CONFIG
# and OUT, e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_dram.sh
#
# For every pipeline and every batch size N of the benchmarks, profiles one execution (the NVTX range "measure" of the
# --ncu-target mode of the benchmarks) twice: with the caches flushed before every kernel (--cache-control all, cold) and without
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
# The NPPDx configuration (tile and block) the NPPDx benchmark chose for every N, from its run without a mode
NPPDX_CONFIG=${NPPDX_CONFIG:-$ROOT/results/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv}
OUT=${OUT:-$ROOT/results/ncu}
OWNER=${SUDO_USER:-$(id -un)}
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

# The batch sizes, from the benchmarks themselves (common/bench_geometry.h)
BATCHES=$("$PAPER_BENCH" --batches) || exit 1
if [ "$BATCHES" != "$("$NPPDX_BENCH" --batches)" ]; then
    echo "$PAPER_BENCH and $NPPDX_BENCH have different batch sizes: build both from the same checkout" >&2
    exit 1
fi
if [ ! -f "$NPPDX_CONFIG" ]; then
    echo "No NPPDx configuration file $NPPDX_CONFIG: run benchmark_figure1_bvf first, or set NPPDX_CONFIG" >&2
    exit 1
fi
nppdx_config() { # <N>: the tile and requested block chosen for N (columns Batch size, Tile, Requested BlockDim)
    awk -F', *' -v n="$1" '$1 == n { print $2, $3 }' "$NPPDX_CONFIG"
}

mkdir -p "$OUT"
LOG=$OUT/ncu_dram.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <N> <cache control> [<NPPDx tile> <NPPDx block>]
    local name=$1 bin=$2 impl=$3 n=$4 cache=$5
    shift 5
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
        for impl in opencv_cuda npp fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf; do
            profile "${impl}_N${n}_cache-${cache}" "$PAPER_BENCH" "$impl" "$n" "$cache"
        done
        # shellcheck disable=SC2046  # the tile and the block are two arguments
        profile "nppdxrepo_nppdx_N${n}_cache-${cache}" "$NPPDX_BENCH" nppdx "$n" "$cache" $(nppdx_config "$n")
        profile "nppdxrepo_fkl_bvf_N${n}_cache-${cache}" "$NPPDX_BENCH" fkl_bvf "$n" "$cache"
    done
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
