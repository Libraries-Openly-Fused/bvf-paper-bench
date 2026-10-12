# Sourced by ncu_dram.sh and ncu_l2.sh: stops unless this user may read the GPU performance counters, sets the paths
# (NCU, PAPER_BENCH, NPPDX_BENCH, NPPDX_NPP_BENCH, RESULTS from BVF_RESULTS, NPPDX_CONFIG, NPPDX_NPP_CONFIG, OWNER),
# checks that the three benchmarks and the NPPDx configuration files are of one workload, and sets BATCHES (the batch
# sizes of the benchmarks) and nppdx_config <configuration CSV> <N>.

# Root is only needed while the driver restricts the GPU performance counters to admin users
# (RmProfilingAdminOnly=1, the default; the nvidia module option NVreg_RestrictProfilingToAdminUsers=0 opens them)
if [ "$(id -u)" -ne 0 ] && ! grep -q "RmProfilingAdminOnly: 0" /proc/driver/nvidia/params 2>/dev/null; then
    echo "The GPU performance counters are restricted to admin users (RmProfilingAdminOnly=1):" \
         "run this script with sudo: sudo $0" >&2
    exit 1
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(dirname "$REPO")
NCU=${NCU:-/usr/local/cuda/bin/ncu}
PAPER_BENCH=${PAPER_BENCH:-$REPO/build/bvf_bench}
NPPDX_BENCH=${NPPDX_BENCH:-$REPO/build-nppdx/benchmark_figure1_bvf}
NPPDX_NPP_BENCH=${NPPDX_NPP_BENCH:-$REPO/build-nppdx/benchmark_figure1_nppdx_npp}
# The NPPDx configuration (tile and block) each NPPDx benchmark chose for every N, from its run without a mode
RESULTS=${BVF_RESULTS:-$ROOT/results}
NPPDX_CONFIG=${NPPDX_CONFIG:-$RESULTS/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv}
NPPDX_NPP_CONFIG=${NPPDX_NPP_CONFIG:-$RESULTS/nppdx_npp/benchmark_figure1_nppdx_npp_nppdx_config_GPU_NVIDIA.csv}
OWNER=${SUDO_USER:-$(id -un)}

# The workload (frame, sizes, batch sizes, crop positions of common/bench_geometry.h) must be the same in the three
# benchmarks and in the runs that chose the NPPDx configurations; the batch sizes come from the benchmarks themselves
WORKLOAD=$("$PAPER_BENCH" --workload) || exit 1
for bench in "$NPPDX_BENCH" "$NPPDX_NPP_BENCH"; do
    if [ "$WORKLOAD" != "$("$bench" --workload)" ]; then
        echo "$PAPER_BENCH and $bench were built for different workloads: configure both with the same" \
             "BENCH_CROP_W, BENCH_CROP_H, BENCH_RESIZE_W and BENCH_RESIZE_H" >&2
        exit 1
    fi
done
BATCHES=$("$PAPER_BENCH" --batches) || exit 1
check_config() { # <configuration CSV> <benchmark name>: it exists and comes from a run with this workload
    if [ ! -f "$1" ]; then
        echo "No NPPDx configuration file $1: run $2 first, or set the variable of its path" >&2
        exit 1
    fi
    local platform
    platform=$(dirname "$1")/$2_platform.txt
    if [ ! -f "$platform" ] || ! grep -qF "Workload: $WORKLOAD" "$platform"; then
        echo "$1 does not come from a run with this workload (no matching Workload line in $platform)" >&2
        exit 1
    fi
}
check_config "$NPPDX_CONFIG" benchmark_figure1_bvf
check_config "$NPPDX_NPP_CONFIG" benchmark_figure1_nppdx_npp
nppdx_config() { # <configuration CSV> <N>: the tile and requested block chosen for N (columns Batch size, Tile,
                 # Requested BlockDim)
    awk -F', *' -v n="$2" '$1 == n { print $2, $3 }' "$1"
}
