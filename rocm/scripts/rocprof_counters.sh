#!/bin/bash
# Experiment 2, memory traffic: hardware counters (rocprofv3 --pmc) of one execution of every pipeline, with the
# caches flushed before every GPU operation, after three warm-up executions (rocm_bench --rocprof-target), for every
# batch size N of the benchmark (rocm_bench --batches, common/bench_geometry.h). One rocprofv3 run per counter, so that
# no counter needs a second hardware pass. The counters of the dispatches after the marker kernel are summed,
# without the flush kernels (summarize_counters.py).
#   rocm/scripts/rocprof_counters.sh <output dir> [<rocm_bench executable>]
#   rocm/scripts/rocprof_counters.sh --check <scratch dir> [<rocm_bench executable>]
# --check only tests whether the hardware counters count here (exit status 0; 3 if they read 0, see below).
#
# COUNTERS (default "FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum"): the counters to collect, as listed by
# rocprofv3 -L for the GPU (rocprofiler-sdk 1.4.1 has no WRITE_SIZE for gfx12; summarize_counters.py derives it).
# MAX_REPEATS (default 3): rounds of the flush leak guard (summarize_counters.py --outliers): a profile whose measured
# execution exceeds the smallest of its three warm-up executions by more than 2 % (and 256 KiB of traffic) is collected
# again; the rejected one is kept in raw_rejected/<label>/<counter>.<attempt>.
#
# Counter collection needs the GPU's performance counters: bare metal (/dev/kfd, membership of the render group; not
# WSL2), and on gfx1200 a profiling performance level: at "auto" the GPU's clock and power gating stay on and the GL2C
# counters read 0 (sudo amd-smi set -l STABLE_STD). Before collecting, the script checks that they count: the
# FETCH_SIZE of the cache flush kernels of one rocm_bench --rocprof-target run must be at least half the flush buffer.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SUMMARIZE="$REPO/rocm/scripts/summarize_counters.py"
CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=1
    shift
fi
OUT=$(realpath -m "${1:?usage: rocprof_counters.sh [--check] <output dir> [<rocm_bench>]}")
BENCH=$(realpath -m "${2:-$REPO/build-rocm/rocm_bench}")
. "$REPO/rocm/scripts/env.sh"
COUNTERS=${COUNTERS:-"FETCH_SIZE GL2C_EA_WRREQ_sum GL2C_EA_WRREQ_64B_sum"}
MAX_REPEATS=${MAX_REPEATS:-3}
IMPLS="rpp_frame rpp_crop fkl_vf_hf fkl_vf_hf_float3_frame fkl_bvf"
mkdir -p "$OUT"
# rocprofv3 writes .rocprofv3/<pid>-counter_values.dat into its working directory: it runs from a scratch directory of
# $OUT, removed on exit, instead of the caller's (the repository root, usually)
PROFILER_CWD=$OUT/.rocprofv3_cwd
mkdir -p "$PROFILER_CWD"
trap 'rm -rf "$PROFILER_CWD"' EXIT

perf_level() {
    amd-smi metric -g 0 --perf-level 2>/dev/null | sed -n 's/^ *PERF_LEVEL: *//p' | head -1
}

# Whether the counters count: FETCH_SIZE of the flush kernels of one FKL BVF target run, N = 1 (exit status 3 if not)
check_counters() {
    local dir=$OUT/counter_check
    rm -rf "$dir" "$OUT/counter_check.log"
    if ! (cd "$PROFILER_CWD" && rocprofv3 --pmc FETCH_SIZE --output-format csv -d "$dir" -- \
            "$BENCH" "$dir/target" --rocprof-target fkl_bvf 1) > "$OUT/counter_check.log" 2>&1; then
        tail -20 "$OUT/counter_check.log"
        echo "rocprofv3 failed in the counter check (log: $OUT/counter_check.log)"
        return 1
    fi
    local status=0
    python3 "$SUMMARIZE" --check-open "$dir" "$OUT/counter_check.log" || status=$?
    if [ $status = 3 ]; then
        echo "The hardware counters read 0 here (performance level: $(perf_level)): on gfx1200 they only count at a" \
             "profiling performance level, e.g. sudo amd-smi set -l STABLE_STD"
    fi
    return $status
}

if [ $CHECK_ONLY = 1 ]; then
    status=0
    check_counters || status=$?
    exit $status
fi

# rpp_frame and rpp_crop are the RPP pipeline chosen by the GPU benchmark's selection pass (the "chosen" row of
# gpu/rpp_selection.csv), with the NV12 -> RGB conversion of the full frame and of only the crops
[ -n "${BENCH_RPP_VARIANT:-}" ] || { echo "set BENCH_RPP_VARIANT to the chosen variant of gpu/rpp_selection.csv"; exit 1; }
export BENCH_RPP_VARIANT
# The check first: if it fails, an earlier collection in $OUT stays as it was
set +e
CHECK_OUT=$(check_counters 2>&1)
status=$?
set -e
echo "$CHECK_OUT"
[ "$status" = 0 ] ||
    { echo "rocprof_counters.sh: STOP: the counter check failed (exit status $status); $OUT left unchanged"; exit "$status"; }
rm -rf "$OUT/raw" "$OUT/raw_rejected" "$OUT/traffic.csv" "$OUT/traffic_repeats.csv"
mkdir -p "$OUT/raw"
{
    echo "rocprof_counters.sh started $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Performance level: $(perf_level)"
    echo "Counters: $COUNTERS; rocprofv3 $(rocprofv3 --version 2>&1 | sed -n 's/^ *version: *//p' | head -1)"
    echo "Target: $BENCH, BENCH_RPP_VARIANT=$BENCH_RPP_VARIANT"
    echo "$CHECK_OUT"
} > "$OUT/platform.txt"

collect_one() { # <label> <counter> <rocm_bench target arguments...>: one rocprofv3 run, in <label>/<counter>
    local label=$1 counter=$2; shift 2
    rm -rf "$OUT/raw/$label/$counter"
    (cd "$PROFILER_CWD" && rocprofv3 --pmc "$counter" --output-format csv -d "$OUT/raw/$label/$counter" -- \
        "$BENCH" "$OUT/raw/$label/$counter/target" "$@") > "$OUT/raw/$label.$counter.log" 2>&1 ||
        { tail -20 "$OUT/raw/$label.$counter.log"; echo "rocprofv3 failed for $label $counter"; exit 1; }
}

BATCHES=$("$BENCH" --batches) || { echo "$BENCH --batches failed"; exit 1; }
for N in $BATCHES; do
    for impl in $IMPLS; do
        for counter in $COUNTERS; do
            collect_one "${impl}_N${N}" "$counter" --rocprof-target "$impl" "$N"
        done
        echo "${impl}_N${N}"
    done
done

# Flush leak guard: collect the suspected profiles again ("<label> <counter>" lines), keeping the rejected ones
for round in $(seq 1 "$MAX_REPEATS"); do
    SUSPECTS=$(python3 "$SUMMARIZE" --outliers "$OUT")
    [ -n "$SUSPECTS" ] || break
    while read -r label counter; do
        mkdir -p "$OUT/raw_rejected/$label"
        attempt=$(( $(find "$OUT/raw_rejected/$label" -mindepth 1 -maxdepth 1 -type d -name "$counter.*" | wc -l) + 1 ))
        mv "$OUT/raw/$label/$counter" "$OUT/raw_rejected/$label/$counter.$attempt"
        mv "$OUT/raw/$label.$counter.log" "$OUT/raw_rejected/$label/$counter.$attempt.log"
        collect_one "$label" "$counter" --rocprof-target "${label%_N*}" "${label##*_N}" < /dev/null
        echo "$label: $counter collected again (flush traffic suspected in profile $attempt, round $round)" |
            tee -a "$OUT/platform.txt"
    done <<< "$SUSPECTS"
done
python3 "$SUMMARIZE" "$OUT"
