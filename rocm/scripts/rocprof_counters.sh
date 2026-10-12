#!/bin/bash
# Experiment 2, memory traffic: hardware counters (rocprofv3 --pmc) of one execution of every pipeline, with the
# caches flushed before every GPU operation, after three warm-up executions (rocm_bench --rocprof-target), for every
# batch size N of the benchmark (rocm_bench --batches, common/bench_geometry.h). One rocprofv3 run per counter, so that
# no counter needs a second hardware pass. The counters of the dispatches after the marker kernel are summed,
# without the flush kernels (summarize_counters.py).
#   rocm/scripts/rocprof_counters.sh <output dir> [<rocm_bench executable>]
# COUNTERS (default "FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum"): the counters to collect, as listed by
# rocprofv3 -L for the GPU (rocprofiler-sdk 1.4.1 has no WRITE_SIZE for gfx12; summarize_counters.py derives it).
# Counter collection needs the GPU's performance counters (bare metal: /dev/kfd and membership of the render group;
# not available on WSL2).
set -euo pipefail
OUT=$(realpath -m "${1:?usage: rocprof_counters.sh <output dir> [<rocm_bench>]}")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BENCH=${2:-$REPO/build-rocm/rocm_bench}
. "$REPO/rocm/scripts/env.sh"
COUNTERS=${COUNTERS:-"FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum"}
# rpp is the RPP pipeline chosen by the GPU benchmark's selection pass (the "chosen" row of gpu/rpp_selection.csv)
[ -n "${BENCH_RPP_VARIANT:-}" ] || { echo "set BENCH_RPP_VARIANT to the chosen variant of gpu/rpp_selection.csv"; exit 1; }
export BENCH_RPP_VARIANT
IMPLS="rpp fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf"
mkdir -p "$OUT/raw"

collect() { # <label> <rocm_bench target arguments...>: one rocprofv3 run per counter, in <label>/<counter>
    local label=$1; shift
    rm -rf "$OUT/raw/$label"
    for counter in $COUNTERS; do
        rocprofv3 --pmc "$counter" --output-format csv -d "$OUT/raw/$label/$counter" -- \
            "$BENCH" "$OUT/raw/$label/$counter/target" "$@" > "$OUT/raw/$label.$counter.log" 2>&1 ||
            { tail -20 "$OUT/raw/$label.$counter.log"; echo "rocprofv3 failed for $label $counter"; exit 1; }
    done
    echo "$label"
}

BATCHES=$("$BENCH" --batches) || { echo "$BENCH --batches failed"; exit 1; }
for N in $BATCHES; do
    for impl in $IMPLS; do
        collect "${impl}_N${N}" --rocprof-target "$impl" "$N"
    done
done
python3 "$REPO/rocm/scripts/summarize_counters.py" "$OUT"
