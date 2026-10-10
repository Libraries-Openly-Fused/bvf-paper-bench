#!/usr/bin/env bash
# Memory traffic between the SMs and the L2 cache of every Figure 1 pipeline with Nsight Compute: every byte the
# kernels read or write, whether it then stays in L2 or goes to DRAM. Needs root for the GPU performance counters:
#
#     sudo /home/oscar/bvf-bench/paper-bench/scripts/ncu_l2.sh
#
# For every pipeline and N in {1, 16, 100}, profiles one execution (the NVTX range "measure" of the --ncu-target mode
# of the benchmarks) with the caches flushed before every kernel. Reports go to results/ncu_l2 and are given back to
# the user who ran sudo. Nothing on the system is changed. Summarize afterwards, without sudo, with
# ncu_summarize.py l2.
set -u

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this script with sudo: sudo $0" >&2
    exit 1
fi

NCU=/usr/local/cuda-13.3/bin/ncu
ROOT=/home/oscar/bvf-bench
PAPER_BENCH=$ROOT/paper-bench/build/bvf_bench
NPPDX_BENCH=$ROOT/FKL-vs-NPPDx/build/benchmark_figure1_bvf
OUT=$ROOT/results/ncu_l2
OWNER=${SUDO_USER:-oscar}
METRICS=lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

mkdir -p "$OUT"
LOG=$OUT/ncu_l2.log
: > "$LOG"
failures=0
total=0

profile() { # <report name> <benchmark binary> <implementation> <N>
    local name=$1 bin=$2 impl=$3 n=$4
    total=$((total + 1))
    echo "[$total] $name" | tee -a "$LOG"
    if ! "$NCU" --nvtx --nvtx-include "measure/" --metrics "$METRICS" --cache-control all \
            --force-overwrite --export "$OUT/$name" "$bin" "$OUT/scratch" --ncu-target "$impl" "$n" >> "$LOG" 2>&1; then
        echo "    FAILED, see $LOG" | tee -a "$LOG"
        failures=$((failures + 1))
    fi
}

for n in 1 16 100; do
    for impl in opencv_cuda npp fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf; do
        profile "${impl}_N${n}_cache-all" "$PAPER_BENCH" "$impl" "$n"
    done
    for impl in nppdx fkl_bvf; do
        profile "nppdxrepo_${impl}_N${n}_cache-all" "$NPPDX_BENCH" "$impl" "$n"
    done
done

rm -rf "$OUT/scratch"
chown -R "$OWNER": "$OUT"
echo "Done: $((total - failures)) of $total profiles written to $OUT ($failures failed)."
