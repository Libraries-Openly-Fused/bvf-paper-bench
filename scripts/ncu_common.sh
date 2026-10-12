# Sourced by ncu_dram.sh and ncu_l2.sh: stops unless this user may read the GPU performance counters, sets the paths,
# checks that the benchmarks and the NPPDx configuration files are of one workload, sets BATCHES (the batch sizes of the
# benchmarks), and defines profile_pipelines <N>, which calls the script's profile function for every pipeline:
#     profile <report name> <benchmark binary> <implementation> <N> [<NPPDx tile> <NPPDx block>]
# with report names <benchmark folder>__<implementation>_N<N>.
#
# Paths: NCU, the benchmarks OPENCV_CUDA_BENCH, NPP_BENCH, ABLATION_BENCH, NPPDX_BENCH and NPPDX_NPP_BENCH (default
# build-<name>/ of this repository), BVF_RESULTS (the results folder of the workload, default ../results, with the runs
# of the FKL vs NPPDx benchmarks in fkl-vs-nppdx/), NPPDX_CONFIG and NPPDX_NPP_CONFIG (their configuration files).

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
OPENCV_CUDA_BENCH=${OPENCV_CUDA_BENCH:-$REPO/build-opencv-cuda/benchmark_opencv_cuda}
NPP_BENCH=${NPP_BENCH:-$REPO/build-npp/benchmark_npp}
ABLATION_BENCH=${ABLATION_BENCH:-$REPO/build-ablation/benchmark_fkl_ablation}
NPPDX_BENCH=${NPPDX_BENCH:-$REPO/build-nppdx/benchmark_nppdx}
NPPDX_NPP_BENCH=${NPPDX_NPP_BENCH:-$REPO/build-nppdx/benchmark_nppdx_npp}
# The NPPDx configuration (tile and block) each NPPDx benchmark chose for every N, from its run without a mode
RESULTS=${BVF_RESULTS:-$ROOT/results}
NPPDX_CONFIG=${NPPDX_CONFIG:-$RESULTS/fkl-vs-nppdx/benchmark_nppdx_nppdx_config_GPU_NVIDIA.csv}
NPPDX_NPP_CONFIG=${NPPDX_NPP_CONFIG:-$RESULTS/fkl-vs-nppdx/benchmark_nppdx_npp_nppdx_config_GPU_NVIDIA.csv}
OWNER=${SUDO_USER:-$(id -un)}

# The workload (frame, sizes, batch sizes, crop positions of common/bench_geometry.h) must be the same in every
# benchmark and in the runs that chose the NPPDx configurations; the batch sizes come from the benchmarks themselves
WORKLOAD=$("$ABLATION_BENCH" --workload) || exit 1
for bench in "$OPENCV_CUDA_BENCH" "$NPP_BENCH" "$NPPDX_BENCH" "$NPPDX_NPP_BENCH"; do
    if [ "$WORKLOAD" != "$("$bench" --workload)" ]; then
        echo "$ABLATION_BENCH and $bench were built for different workloads: configure both with the same" \
             "BENCH_CROP_W, BENCH_CROP_H, BENCH_RESIZE_W and BENCH_RESIZE_H" >&2
        exit 1
    fi
done
BATCHES=$("$ABLATION_BENCH" --batches) || exit 1
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
check_config "$NPPDX_CONFIG" benchmark_nppdx
check_config "$NPPDX_NPP_CONFIG" benchmark_nppdx_npp
nppdx_config() { # <configuration CSV> <N>: the tile and requested block chosen for N (columns Batch size, Tile,
                 # Requested BlockDim)
    awk -F', *' -v n="$2" '$1 == n { print $2, $3 }' "$1"
}

profile_pipelines() { # <N>: every pipeline of every benchmark at batch size N (FKL BVF once from fkl-ablation and once
                      # from fkl-vs-nppdx, whose programs are built with NPPDx's options)
    local n=$1
    profile "fkl-vs-opencv-cuda__opencv_cuda_N$n" "$OPENCV_CUDA_BENCH" opencv_cuda "$n"
    profile "fkl-vs-npp__npp_N$n" "$NPP_BENCH" npp "$n"
    profile "fkl-ablation__fkl_vf_hf_N$n" "$ABLATION_BENCH" fkl_vf_hf "$n"
    profile "fkl-ablation__fkl_bvf_N$n" "$ABLATION_BENCH" fkl_bvf "$n"
    # shellcheck disable=SC2046  # the tile and the block are two arguments
    profile "fkl-vs-nppdx__nppdx_N$n" "$NPPDX_BENCH" nppdx "$n" $(nppdx_config "$NPPDX_CONFIG" "$n")
    # shellcheck disable=SC2046
    profile "fkl-vs-nppdx__nppdx_npp_N$n" "$NPPDX_NPP_BENCH" nppdx_npp "$n" $(nppdx_config "$NPPDX_NPP_CONFIG" "$n")
    profile "fkl-vs-nppdx__fkl_bvf_N$n" "$NPPDX_BENCH" fkl_bvf "$n"
}
