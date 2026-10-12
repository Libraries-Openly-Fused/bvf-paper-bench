#!/usr/bin/env bash
# Memory traffic between the SMs and the L2 cache of every Figure 1 pipeline with Nsight Compute: every byte the
# kernels read or write, whether it then stays in L2 or goes to DRAM. Needs root for the GPU performance counters:
#
#     sudo scripts/ncu_l2.sh
#
# Paths default to the layout described in README.md (the build and build-nppdx folders of this repository, and
# results/ next to it) and can be overridden with the environment variables NCU, PAPER_BENCH, NPPDX_BENCH, BVF_RESULTS
# (the results folder of one workload: the NPPDx run in nppdx/, the reports in ncu_l2/), NPPDX_CONFIG and OUT, e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_l2.sh
# or, for benchmarks built with other BENCH_CROP_*/BENCH_RESIZE_* options (see README.md),
#     sudo PAPER_BENCH=build-20x10/bvf_bench NPPDX_BENCH=build-nppdx-20x10/benchmark_figure1_bvf \
#          BVF_RESULTS=../results/20x10 scripts/ncu_l2.sh
#
# For every pipeline and every batch size N of the benchmarks, profiles one execution (the NVTX range "measure" of the
# --ncu-target mode of the benchmarks) with the caches flushed before every kernel. Reports go to results/ncu_l2 and are given back to
# the user who ran sudo. Nothing on the system is changed. Summarize afterwards, without sudo, with
# ncu_summarize.py l2.
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
RESULTS=${BVF_RESULTS:-$ROOT/results}
NPPDX_CONFIG=${NPPDX_CONFIG:-$RESULTS/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv}
OUT=${OUT:-$RESULTS/ncu_l2}
OWNER=${SUDO_USER:-$(id -un)}
METRICS=lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

# The workload (frame, sizes, batch sizes, crop positions of common/bench_geometry.h) must be the same in both
# benchmarks and in the run that chose the NPPDx configurations; the batch sizes come from the benchmarks themselves
WORKLOAD=$("$PAPER_BENCH" --workload) || exit 1
if [ "$WORKLOAD" != "$("$NPPDX_BENCH" --workload)" ]; then
    echo "$PAPER_BENCH and $NPPDX_BENCH were built for different workloads: configure both with the same" \
         "BENCH_CROP_W, BENCH_CROP_H, BENCH_RESIZE_W and BENCH_RESIZE_H" >&2
    exit 1
fi
BATCHES=$("$PAPER_BENCH" --batches) || exit 1
if [ ! -f "$NPPDX_CONFIG" ]; then
    echo "No NPPDx configuration file $NPPDX_CONFIG: run benchmark_figure1_bvf first, or set NPPDX_CONFIG" >&2
    exit 1
fi
NPPDX_PLATFORM=$(dirname "$NPPDX_CONFIG")/benchmark_figure1_bvf_platform.txt
if [ ! -f "$NPPDX_PLATFORM" ] || ! grep -qF "Workload: $WORKLOAD" "$NPPDX_PLATFORM"; then
    echo "$NPPDX_CONFIG does not come from a run with this workload (no matching Workload line in $NPPDX_PLATFORM)" >&2
    exit 1
fi
nppdx_config() { # <N>: the tile and requested block chosen for N (columns Batch size, Tile, Requested BlockDim)
    awk -F', *' -v n="$1" '$1 == n { print $2, $3 }' "$NPPDX_CONFIG"
}

mkdir -p "$OUT"
LOG=$OUT/ncu_l2.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <N> [<NPPDx tile> <NPPDx block>]
    local name=$1 bin=$2 impl=$3 n=$4
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
    for impl in opencv_cuda npp fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf; do
        profile "${impl}_N${n}_cache-all" "$PAPER_BENCH" "$impl" "$n"
    done
    # shellcheck disable=SC2046  # the tile and the block are two arguments
    profile "nppdxrepo_nppdx_N${n}_cache-all" "$NPPDX_BENCH" nppdx "$n" $(nppdx_config "$n")
    profile "nppdxrepo_fkl_bvf_N${n}_cache-all" "$NPPDX_BENCH" fkl_bvf "$n"
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
