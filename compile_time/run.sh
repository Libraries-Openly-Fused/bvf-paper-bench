#!/usr/bin/env bash
# Compile-time benchmark: compiles one source once per (IMPL, BATCH) translation unit with the exact nvcc command line of
# a benchmark target, REPS times after one discarded warm-up, and records the wall, user and sys time and the max RSS
# (/usr/bin/time), nvcc's per-phase times (--time) and, from one extra compile, the kernels' resource usage and the code
# sizes. One nvcc at a time, objects on tmpfs, no ccache.
#
#     compile_time/run.sh <results directory>                 # src/bvf_bench.cu's command, compile_time/pipeline.cu
#     SET=nppdx compile_time/run.sh <results directory>       # fkl-vs-nppdx: benchmark_figure1_bvf's command and
#                                                             # fkl-vs-nppdx/benchmarks/compile_time_pipeline.cu
#
# BUILD is the CMake build folder of the benchmark target, configured with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON (default
# build for SET=paper, build-nppdx for SET=nppdx); the geometry is the one it was configured with. REPS (default 10)
# sets the timed compiles per TU. The script waits until the 1 minute load average is below MAX_LOAD (default 0.5)
# before starting. SET=nppdx compiles the NPPDx kernel of every N in the configuration benchmark_figure1_bvf chose for
# it: NPPDX_CONFIG is the benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv of its run (default
# ../results/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv, or nppdx/ of BVF_RESULTS).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:?results directory}" && cd "$1" && pwd)
SET=${SET:-paper}
REPS=${REPS:-10}
MAX_LOAD=${MAX_LOAD:-0.5}
WORK=/dev/shm/bvf-compile-time-$SET
case "$SET" in
  paper)
    BUILD=${BUILD:-$REPO/build}
    SOURCE=$REPO/compile_time/pipeline.cu
    ENTRY=src/bvf_bench.cu
    TUS=("HEADERS_FKL 1" "HEADERS_ALL 1" "OPENCV 100" "NPP 100"
         "VFHF 1" "VFHF 4" "VFHF 16" "VFHF 50" "VFHF 100"
         "BVF 1" "BVF 4" "BVF 16" "BVF 50" "BVF 100"
         "BVF_CWD 100")
    ;;
  nppdx)
    BUILD=${BUILD:-$REPO/build-nppdx}
    SOURCE=$REPO/fkl-vs-nppdx/benchmarks/compile_time_pipeline.cu
    ENTRY=fkl-vs-nppdx/benchmarks/benchmark_figure1_bvf.cu
    TUS=("HEADERS 1"
         "NPPDX 1" "NPPDX 4" "NPPDX 16" "NPPDX 50" "NPPDX 100"
         "BVF 1" "BVF 4" "BVF 16" "BVF 50" "BVF 100")
    NPPDX_CONFIG=${NPPDX_CONFIG:-${BVF_RESULTS:-$(dirname "$REPO")/results}/nppdx/benchmark_figure1_bvf_nppdx_config_GPU_NVIDIA.csv}
    if [ ! -f "$NPPDX_CONFIG" ]; then
      echo "No NPPDx configuration file $NPPDX_CONFIG: run benchmark_figure1_bvf first, or set NPPDX_CONFIG" >&2
      exit 1
    fi
    # The configurations must come from a run with the workload of this build (common/bench_geometry.h)
    WORKLOAD=$("$BUILD/benchmark_figure1_bvf" --workload)
    NPPDX_PLATFORM=$(dirname "$NPPDX_CONFIG")/benchmark_figure1_bvf_platform.txt
    if [ ! -f "$NPPDX_PLATFORM" ] || ! grep -qF "Workload: $WORKLOAD" "$NPPDX_PLATFORM"; then
      echo "$NPPDX_CONFIG does not come from a run with the workload of $BUILD (no matching Workload line in" \
           "$NPPDX_PLATFORM)" >&2
      exit 1
    fi
    ;;
  *) echo "SET must be paper or nppdx" >&2; exit 1 ;;
esac
if [ -n "${ONLY:-}" ]; then  # ONLY="BVF 100;VFHF 100" restricts the TUs (for checks; not for results)
  IFS=';' read -r -a TUS <<< "$ONLY"
fi
rm -rf "$WORK"
mkdir -p "$WORK"

# 1. The nvcc command of ENTRY from compile_commands.json, without its input, output and dependency file options, and
#    with its --options-file (the include paths) expanded: every other flag and definition is kept as it is.
python3 - "$BUILD/compile_commands.json" "$ENTRY" > "$WORK/nvcc_args" <<'EOF'
import json, os, shlex, sys
commands, entry = json.load(open(sys.argv[1])), sys.argv[2]
e = next(e for e in commands if e["file"].endswith(entry))
args = e["arguments"] if "arguments" in e else shlex.split(e["command"])
out, skip = [], False
for i, a in enumerate(args):
    if skip:
        skip = False
        continue
    if a in ("-o", "-MF", "-MT", "-MQ"):
        skip = True
        continue
    if a in ("-c", "-MD", "-MMD") or a == e["file"] or a.endswith(entry):
        continue
    if a == "--options-file":
        rsp = args[i + 1]
        rsp = rsp if os.path.isabs(rsp) else os.path.join(e["directory"], rsp)
        out += shlex.split(open(rsp).read())
        skip = True
        continue
    out.append(a)
print("\n".join(out))
EOF
mapfile -t NVCC < "$WORK/nvcc_args"
printf '%q ' "${NVCC[@]}" > "$OUT/nvcc_command.txt"
echo >> "$OUT/nvcc_command.txt"
FKL_DIR=$(sed -n 's/^FKL_DIR:PATH=//p' "$BUILD/CMakeCache.txt")
NPPDX_DIR=$(sed -n 's/^NPPDX_DIR:PATH=//p' "$BUILD/CMakeCache.txt")
GEOMETRY=$(for v in BENCH_CROP_W BENCH_CROP_H BENCH_RESIZE_W BENCH_RESIZE_H; do
             echo -n "$v=$(sed -n "s/^$v:STRING=//p" "$BUILD/CMakeCache.txt") "; done)

tu_flags() {  # impl batch: the definitions of one TU; the NPPDx kernel in the configuration chosen for this batch
  echo "-DIMPL_$1 -DCT_BATCH=$2"
  if [ "$1" = NPPDX ]; then
    # Batch size, Tile, Requested BlockDim, ...
    awk -F', *' -v n="$2" '$1 == n { split($2, t, "x"); found = 1
        print "-DCT_NPPDX_TILE_X=" t[1] " -DCT_NPPDX_TILE_Y=" t[2] " -DCT_NPPDX_BLOCK=" $3 }
        END { if (!found) exit 1 }' "$NPPDX_CONFIG" || { echo "No NPPDx configuration for N = $2" >&2; return 1; }
  fi
}

# 2. Wait for an idle machine
for _ in $(seq 60); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN { exit !(l < m) }'; then break; fi
  echo "load average $load >= $MAX_LOAD, waiting" >&2
  sleep 10
done
if ! awk -v l="$(cut -d' ' -f1 /proc/loadavg)" -v m="$MAX_LOAD" 'BEGIN { exit !(l < m) }'; then
  echo "The machine is not idle (load average $(cut -d' ' -f1 /proc/loadavg)), not measuring" >&2
  exit 1
fi
LOAD_BEFORE=$(cat /proc/loadavg)
START=$(date -Is)

compile() {  # impl batch rep: one timed compile, wall/user/sys/maxrss appended to compile_time.csv
  local impl=$1 batch=$2 rep=$3
  local tag="${impl}_${batch}" flags
  flags=$(tu_flags "$impl" "$batch")
  # shellcheck disable=SC2086  # flags holds several definitions
  /usr/bin/time -f "%e,%U,%S,%M" -o "$WORK/time.txt" \
    "${NVCC[@]}" $flags --time "$OUT/phases_$tag.csv" -c "$SOURCE" -o "$WORK/$tag.o"
  echo "$impl,$batch,$rep,$(cat "$WORK/time.txt")" >> "$OUT/compile_time.csv"
}

busy_jiffies() {  # CPU time the whole machine has used since boot (user, nice, system, irq, softirq, steal)
  awk '/^cpu / { print $2 + $3 + $4 + $7 + $8 + $9 }' /proc/stat
}
rm -f "$OUT"/phases_*.csv "$OUT/compile_time.csv" "$OUT/load.csv"
echo "impl,batch,rep,wall_s,user_s,sys_s,maxrss_kb" > "$OUT/compile_time.csv"
# Per round: the machine's busy CPU time, to compare with the user + sys time of the round's nvcc processes (the load
# average mostly shows the benchmark's own nvcc)
echo "rep,busy_jiffies_start,busy_jiffies_end,clock_ticks_per_s,loadavg_1min" > "$OUT/load.csv"
for rep in $(seq 0 "$REPS"); do                    # rep 0: warm-up (page cache), left out of the summary;
  start=$(busy_jiffies)                            # round-robin, so that any drift affects every TU alike
  for tu in "${TUS[@]}"; do compile $tu "$rep"; done
  echo "$rep,$start,$(busy_jiffies),$(getconf CLK_TCK),$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/load.csv"
  echo "round $rep of $REPS done" >&2
done
END=$(date -Is)

# 3. One untimed compile per TU with ptxas' resource usage and the intermediate files kept, for the kernel facts and sizes
for tu in "${TUS[@]}"; do
  set -- $tu
  tag="$1_$2"
  keep="$OUT/keep_$tag"
  rm -rf "$keep"
  mkdir -p "$keep"
  # shellcheck disable=SC2046
  "${NVCC[@]}" $(tu_flags "$1" "$2") -Xptxas -v --keep --keep-dir "$keep" -c "$SOURCE" -o "$keep/$tag.o" \
    > "$keep/ptxas.log" 2>&1
  "$(dirname "${NVCC[0]}")/cuobjdump" --dump-resource-usage "$keep/$tag.o" > "$keep/resource_usage.txt"
  stat -c "%s" "$WORK/$tag.o" > "$keep/object_bytes.txt"   # the object of the last timed compile
  # The kept intermediates are large; keep only what the summary reads
  find "$keep" -maxdepth 1 -type f \( -name '*.ii' -o -name '*.cudafe1.*' -o -name '*.module_id' -o -name '*.fatbin*' \
       -o -name '*.reg.c' -o -name '*.stub.c' \) -delete
  for f in "$keep"/*.ptx "$keep"/*.cubin; do [ -f "$f" ] && echo "$(stat -c %s "$f") $(basename "$f")"; done \
    > "$keep/sizes.txt"
  rm -f "$keep"/*.ptx
done

# 4. Platform
{
  echo "date: $START to $END"
  echo "load average before start: $LOAD_BEFORE; machine CPU time outside the benchmark's nvcc per round in the summary"
  echo "nvcc: $("${NVCC[0]}" --version | tail -2 | tr '\n' ' ')"
  echo "host compiler: $(gcc --version | head -1)"
  echo "CPU: $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) hardware threads, governor:" \
       "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "OS: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
  echo "FKL: $FKL_DIR at $(git -C "$FKL_DIR" rev-parse --short HEAD) ($(git -C "$FKL_DIR" rev-parse --abbrev-ref HEAD))"
  if [ -n "$NPPDX_DIR" ]; then
    echo "NPPDx: $NPPDX_DIR at $(git -C "$NPPDX_DIR" describe --always --tags)"
  fi
  echo "bvf-paper-bench: $(git -C "$REPO" rev-parse --short HEAD)$([ -z "$(git -C "$REPO" status --porcelain)" ] || echo ' (with uncommitted or untracked changes)')"
  echo "source: ${SOURCE#$REPO/}, nvcc command of ${ENTRY} from $BUILD/compile_commands.json"
  echo "geometry: $GEOMETRY(common/bench_geometry.h)"
  if [ "$SET" = nppdx ]; then
    for tu in "${TUS[@]}"; do
      set -- $tu
      [ "$1" = NPPDX ] && echo "NPPDx configuration for N = $2: $(tu_flags "$1" "$2" | tail -1) (from $NPPDX_CONFIG)"
    done
  fi
  echo "protocol: $REPS timed compiles per TU after 1 discarded warm-up, round-robin over the TUs, one nvcc at a time" \
       "(no -j, no ccache), -c only, objects on tmpfs ($WORK); /usr/bin/time around nvcc and nvcc --time in the same" \
       "compile; one more untimed compile per TU with -Xptxas -v --keep for the kernel facts and sizes"
  echo "nvcc command: $(cat "$OUT/nvcc_command.txt")"
} > "$OUT/compile_time_platform.txt"
rm -rf "$WORK"
python3 "$REPO/compile_time/summarize.py" "$OUT"
