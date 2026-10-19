# Sourced by ncu_dram.sh and ncu_l2.sh: stops unless this user may read the GPU performance counters, sets the paths,
# checks that the benchmarks are of one workload, sets BATCHES (the batch sizes of the benchmarks) and the batch sizes
# of the warm and SM-L2 profiles, writes the regions of the profiled iteration to ncu_regions.csv in the results folder
# (for the traffic model), and defines profile_pipelines <N>, which calls the script's profile function for every
# pipeline of PIPELINES: OpenCV CUDA, NPP in both forms of its NV12 conversion (npp and npp_crop_nv12), FKL VF+HF and
# FKL BVF, with report names <benchmark folder>__<implementation>_N<N>.
#
# Every profile is one execution on the regions of iteration 0 (the --ncu-target mode of the benchmarks), the same
# regions for every pipeline.
#
# Paths: NCU, the benchmarks OPENCV_CUDA_BENCH, NPP_BENCH and ABLATION_BENCH (default build-<name>/ of this
# repository), BVF_RESULTS (the results folder, default ../results). WARM_BATCHES (default "1 100"; at least one, each
# a batch size of the benchmarks, listed once) are the batch sizes of the warm DRAM profiles and of the SM-L2 profiles;
# the cold DRAM profiles cover every batch size. PIPELINES (default: every one) limits the profiles to the listed
# implementations among opencv_cuda, npp, npp_crop_nv12, fkl_vf_hf and fkl_bvf, e.g. PIPELINES=opencv_cuda with OUT a
# new folder to profile one pipeline again after a change of its program (each script deletes the earlier reports of
# its OUT folder; ncu_summarize.py summarizes one folder).
# Root is only needed while the driver restricts the GPU performance counters to admin users
# (RmProfilingAdminOnly=1, the default; the nvidia module option NVreg_RestrictProfilingToAdminUsers=0 opens them)
if [ "$(id -u)" -ne 0 ] && ! grep -q "RmProfilingAdminOnly: 0" /proc/driver/nvidia/params 2>/dev/null; then
    echo "The GPU performance counters are restricted to admin users (RmProfilingAdminOnly=1):" \
         "run this script with sudo: sudo $0" >&2
    exit 1
fi
ALL_PIPELINES="opencv_cuda npp npp_crop_nv12 fkl_vf_hf fkl_bvf"
PIPELINES=${PIPELINES:-$ALL_PIPELINES}
for pipeline in $PIPELINES; do
    if [[ " $ALL_PIPELINES " != *" $pipeline "* ]]; then
        echo "PIPELINES: $pipeline is not one of the profiled pipelines ($ALL_PIPELINES)" >&2
        exit 1
    fi
done
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(dirname "$REPO")
NCU=${NCU:-/usr/local/cuda/bin/ncu}
OPENCV_CUDA_BENCH=${OPENCV_CUDA_BENCH:-$REPO/build-opencv-cuda/benchmark_opencv_cuda}
NPP_BENCH=${NPP_BENCH:-$REPO/build-npp/benchmark_npp}
ABLATION_BENCH=${ABLATION_BENCH:-$REPO/build-ablation/benchmark_fkl_ablation}
RESULTS=${BVF_RESULTS:-$ROOT/results}
WARM_BATCHES=${WARM_BATCHES:-"1 100"}
OWNER=${SUDO_USER:-$(id -un)}
# The workload (frame, region sizes and positions, batch sizes of common/bench_geometry.h) must be the same in every
# benchmark; the batch sizes come from the benchmarks themselves
WORKLOAD=$("$ABLATION_BENCH" --workload) || exit 1
for bench in "$OPENCV_CUDA_BENCH" "$NPP_BENCH"; do
    if [ "$WORKLOAD" != "$("$bench" --workload)" ]; then
        echo "$ABLATION_BENCH and $bench were built for different workloads: configure both with the same" \
             "BENCH_REGIONS" >&2
        exit 1
    fi
done
BATCHES=$("$ABLATION_BENCH" --batches) || exit 1
# Checked before the scripts delete earlier reports. A batch size listed twice would overwrite its own reports.
warm_seen=" "
for n in $WARM_BATCHES; do
    if [[ " $BATCHES " != *" $n "* ]]; then
        echo "WARM_BATCHES: $n is not one of the batch sizes of the benchmarks ($BATCHES)" >&2
        exit 1
    fi
    if [[ $warm_seen == *" $n "* ]]; then
        echo "WARM_BATCHES: $n is listed twice" >&2
        exit 1
    fi
    warm_seen+="$n "
done
if [ "$warm_seen" = " " ]; then
    echo "WARM_BATCHES is empty: it needs at least one batch size of the benchmarks ($BATCHES)" >&2
    exit 1
fi
mkdir -p "$RESULTS"
"$ABLATION_BENCH" --regions 0 > "$RESULTS/ncu_regions.csv" || exit 1
chown "$OWNER": "$RESULTS/ncu_regions.csv"
profile_pipeline() { # <benchmark folder> <benchmark binary> <implementation> <N>: profiled if PIPELINES lists it
    if [[ " $PIPELINES " == *" $3 "* ]]; then
        profile "$1__$3_N$4" "$2" "$3" "$4"
    fi
}
profile_pipelines() { # <N>: every profiled pipeline of PIPELINES at batch size N
    local n=$1
    profile_pipeline fkl-vs-opencv-cuda "$OPENCV_CUDA_BENCH" opencv_cuda "$n"
    profile_pipeline fkl-vs-npp "$NPP_BENCH" npp "$n"
    profile_pipeline fkl-vs-npp "$NPP_BENCH" npp_crop_nv12 "$n"
    profile_pipeline fkl-ablation "$ABLATION_BENCH" fkl_vf_hf "$n"
    profile_pipeline fkl-ablation "$ABLATION_BENCH" fkl_bvf "$n"
}
