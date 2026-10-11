#!/usr/bin/env bash
# Memory traffic of every pipeline of the region-size sweep with Nsight Compute, in one go: cold DRAM traffic and the
# traffic between the SMs and the L2 cache. Needs root for the GPU performance counters (RmProfilingAdminOnly=1):
#
#     sudo scripts/ncu_region_sweep.sh
#
# Paths default to the layout described in README.md (the build and build-nppdx folders of this repository, and
# results/ next to it) and can be overridden with the environment variables NCU, PAPER_BENCH, NPPDX_BENCH, NPPDX_CONFIGS and
# OUT, e.g.
#     sudo NCU=/usr/local/cuda-13.3/bin/ncu scripts/ncu_region_sweep.sh
#
# For every region case (A to D, N = 16) and pipeline, profiles one execution (the NVTX range "measure" of the
# --region-ncu-target mode of both benchmarks) with the caches flushed before every kernel. NPPDx runs in the tile and
# block configuration its --region-sweep run chose (NPPDX_CONFIGS, the *_nppdx_configs_* CSV of that run). Reports go
# to results/ncu_region and are given back to the user who ran sudo. Nothing on the system is changed. Summarize
# afterwards, without sudo, with ncu_summarize.py region.
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
NPPDX_CONFIGS=${NPPDX_CONFIGS:-$ROOT/results/region_sweep/nppdx/benchmark_figure1_bvf_region_sweep_nppdx_configs_GPU_NVIDIA.csv}
OUT=${OUT:-$ROOT/results/ncu_region}
OWNER=${SUDO_USER:-$(id -un)}
METRICS=lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

if [ ! -f "$NPPDX_CONFIGS" ]; then
    echo "NPPDx configurations not found: $NPPDX_CONFIGS (run build-nppdx/benchmark_figure1_bvf with --region-sweep first)" >&2
    exit 1
fi

mkdir -p "$OUT"
LOG=$OUT/ncu_region_sweep.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <region case> [<NPPDx tile> <NPPDx block>]
    local name=$1 bin=$2 impl=$3 region=$4
    shift 4
    total=$((total + 1))
    echo "[$total] $name $*" | tee -a "$LOG"
    if ! "$NCU" --nvtx --nvtx-include "measure/" --metrics "$METRICS" --cache-control all \
            --force-overwrite --export "$OUT/$name" "$bin" "$OUT/scratch" --region-ncu-target "$impl" "$region" "$@" \
            >> "$LOG" 2>&1; then
        echo "    FAILED, see $LOG" | tee -a "$LOG"
        failures=$((failures + 1))
    fi
}

for region in A B C D; do
    for impl in opencv_cuda npp fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf; do
        profile "region${region}_${impl}_N16_cache-all" "$PAPER_BENCH" "$impl" "$region"
    done
    # Tile and requested block of the configuration NPPDx's sweep chose for this case
    nppdx_config=$(awk -F', *' -v r="$region" '$1 == r && $NF == "yes" { print $3, $4 }' "$NPPDX_CONFIGS")
    if [ -z "$nppdx_config" ]; then
        echo "No chosen NPPDx configuration for case $region in $NPPDX_CONFIGS" | tee -a "$LOG"
        failures=$((failures + 1))
    else
        profile "region${region}_nppdxrepo_nppdx_N16_cache-all" "$NPPDX_BENCH" nppdx "$region" $nppdx_config
    fi
    profile "region${region}_nppdxrepo_fkl_bvf_N16_cache-all" "$NPPDX_BENCH" fkl_bvf "$region"
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
