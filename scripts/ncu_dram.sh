#!/usr/bin/env bash
# DRAM traffic of every Figure 1 pipeline with Nsight Compute. Needs root while the NVIDIA driver restricts the GPU
# performance counters to admin users (RmProfilingAdminOnly=1, the default; without it, run it without sudo):
#
#     sudo scripts/ncu_dram.sh
#
# Paths default to the layout described in README.md (the build and build-nppdx folders of this repository, and
# results/ next to it) and can be overridden with the environment variables NCU, PAPER_BENCH, NPPDX_BENCH,
# NPPDX_NPP_BENCH, BVF_RESULTS (the results folder of one workload: the NPPDx runs in nppdx/ and nppdx_npp/, the reports
# in ncu/), NPPDX_CONFIG, NPPDX_NPP_CONFIG and OUT, e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_dram.sh
# or, for benchmarks built with other BENCH_CROP_*/BENCH_RESIZE_* options (see README.md),
#     sudo PAPER_BENCH=build-20x10/bvf_bench NPPDX_BENCH=build-nppdx-20x10/benchmark_figure1_bvf \
#          NPPDX_NPP_BENCH=build-nppdx-20x10/benchmark_figure1_nppdx_npp \
#          BVF_RESULTS=../results/20x10 scripts/ncu_dram.sh
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
        for impl in opencv_cuda npp fkl_vf_hf fkl_bvf; do
            profile "${impl}_N${n}_cache-${cache}" "$PAPER_BENCH" "$impl" "$n" "$cache"
        done
        # shellcheck disable=SC2046  # the tile and the block are two arguments
        profile "nppdxrepo_nppdx_N${n}_cache-${cache}" "$NPPDX_BENCH" nppdx "$n" "$cache" \
            $(nppdx_config "$NPPDX_CONFIG" "$n")
        # shellcheck disable=SC2046
        profile "nppdxrepo_nppdx_npp_N${n}_cache-${cache}" "$NPPDX_NPP_BENCH" nppdx_npp "$n" "$cache" \
            $(nppdx_config "$NPPDX_NPP_CONFIG" "$n")
        profile "nppdxrepo_fkl_bvf_N${n}_cache-${cache}" "$NPPDX_BENCH" fkl_bvf "$n" "$cache"
    done
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
