#!/usr/bin/env python3
"""Summarizes rocm/scripts/rocprof_counters.sh (Experiment 2, memory traffic).

Every profile (raw/<impl>_N<n>/<counter>, one rocprofv3 run of rocm_bench --rocprof-target) holds three warm-up
executions of one pipeline, measureMarkerKernel, then the measured execution, with a cache flush kernel before every GPU
operation of all four executions.

    summarize_counters.py <dir>
        For every <impl>_N<n> directory of <dir>/raw, the counters of the dispatches of the measured execution (after
        measureMarkerKernel), without the cache flush kernels, summed per counter. The HIP runtime's own blit kernels
        (__amd_rocclr_*, which execute memsets and some copies) are included in the sums, since their memory traffic is
        part of the pipeline's, and counted separately from the pipeline's kernels. Writes <dir>/traffic.csv,
        <dir>/traffic_repeats.csv (the profiles rejected by the flush leak guard, see --outliers) and the notes on the
        counters into <dir>/platform.txt (replacing the notes of an earlier run of this script).
        WRITE_SIZE: rocprofiler-sdk 1.4.1 defines no WRITE_SIZE for gfx12 (its gfx10/gfx11 WRITE_SIZE uses
        GL2C_MC_WRREQ_sum, which gfx12 lacks), so when WRITE_SIZE is not collected it is derived, as
        "WRITE_SIZE (derived)", from GL2C_EA_WRREQ_sum and GL2C_EA_WRREQ_64B_sum with the request size of the GPU's
        architecture (WRITE_REQUEST_BYTES; the gfx10/gfx11 entry is only a fallback for when WRITE_SIZE is not collected).

    summarize_counters.py --outliers <dir>
        The flush leak guard: prints "<impl>_N<n> <counter>" for every profile whose measured execution exceeds the
        smallest of its three warm-up executions (the same dispatches, checked by kernel name, each also with the caches
        flushed) by more than max(LEAK_RELATIVE x that warm-up, LEAK_FLOOR of the counter). The counts of a flush
        kernel occasionally land on the dispatches right after it, which inflates an execution (gfx1200, rocprofiler-sdk
        1.4.1: 5 of 60 profiles of one collection, from +3.6 MiB to +96 MiB of FETCH_SIZE and about +98000
        GL2C_EA_WRREQ_sum requests). rocprof_counters.sh collects those profiles again.

    summarize_counters.py --check-open <rocprofv3 output dir> <its log>
        Whether the hardware counters count on this GPU at its current performance level: the FETCH_SIZE of the flush
        kernels of one rocm_bench --rocprof-target run must be at least OPEN_FRACTION of the flush buffer they read.
        On gfx1200 the GL2C counters (and TA, TCP, the SQ instruction counters) read 0 at the "auto" performance level,
        which leaves the GPU's clock and power gating on, and count at a profiling level (amd-smi set -l STABLE_STD).
        Exit status 0: the counters count; 3: they read 0 (or too little); 1: the run has no flush kernel records.
"""
import csv
import glob
import os
import re
import sys

# Bytes per GL2C_EA write request that is not counted as 64 B, per architecture family (gfx target prefix). gfx10 and
# gfx11: the 32 B / 64 B formula of rocprofiler's gfx10/gfx11 WRITE_SIZE (only a fallback there, see the docstring).
# gfx12: measured on gfx1200 (RX 9060, rocprofiler-sdk 1.4.1) with a kernel that writes 256 MiB: 1042140
# GL2C_EA_WRREQ_sum requests per execution at steady state, none of them GL2C_EA_WRREQ_64B, i.e. 257.6 B per request:
# the requests are full 256 B lines.
WRITE_REQUEST_BYTES = {"gfx10": 32, "gfx11": 32, "gfx12": 256}
# Flush leak guard: an excess of the measured execution over the smallest warm-up of more than LEAK_RELATIVE of it, and
# of more than LEAK_FLOOR in the counter's unit (256 KiB of traffic), is a leak. The executions of one profile agree
# within 0.7 % (GL2C_EA_WRREQ_sum of FKL's output kernel) and within 1 KiB of FETCH_SIZE; the leaks seen were +28 % and
# more. Counters without a floor here are not checked.
LEAK_RELATIVE = 0.02
LEAK_FLOOR = {"FETCH_SIZE": 256.0, "GL2C_EA_WRREQ_sum": 1024.0, "GL2C_EA_WRREQ_64B_sum": 4096.0, "WRITE_SIZE": 256.0}
WARMUPS = 3
# --check-open: a flush kernel reads its whole buffer, which is twice L2 + MALL, so its FETCH_SIZE is at least half of
# it even if the caches kept the other half (measured: exactly the buffer size); 0 when the counters are gated
OPEN_FRACTION = 0.5

FLUSH_RE = re.compile(r"Flushing \d+ MiB \(2 x \(L2 (\d+) KiB \+ MALL (\d+) KiB\)\)")
NOTES_BEGIN = "--- summarize_counters.py notes ---"
NOTES_END = "--- end of summarize_counters.py notes ---"


def read_rows(directory):
    files = glob.glob(os.path.join(directory, "**", "*counter_collection.csv"), recursive=True)
    if not files:
        raise SystemExit(f"no counter_collection.csv under {directory}")
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    for r in rows:
        r["_dispatch"] = int(r["Dispatch_Id"])
    rows.sort(key=lambda r: r["_dispatch"])
    return rows


def is_flush(r):
    return "flushCachesKernel" in r["Kernel_Name"]


def is_marker(r):
    return "measureMarkerKernel" in r["Kernel_Name"]


def summarize(directory):
    """Sums of every counter over the measured dispatches of one rocprofv3 output directory, the dispatch counts, and the
    sums of the same counters over each warm-up execution (None if the warm-ups cannot be matched to the measured
    execution by kernel name)."""
    rows = read_rows(directory)
    markers = sorted({r["_dispatch"] for r in rows if is_marker(r)})
    if len(markers) != 1:
        raise SystemExit(f"{directory}: expected one measureMarkerKernel dispatch, found {markers}")
    marker = markers[0]

    def dispatches(selected):
        """[(dispatch id, kernel name, {counter: value})] in dispatch order"""
        by_id = {}
        for r in selected:
            by_id.setdefault(r["_dispatch"], (r["Kernel_Name"], {}))[1][r["Counter_Name"]] = float(r["Counter_Value"])
        return [(d, name, values) for d, (name, values) in sorted(by_id.items())]

    measured = dispatches([r for r in rows if r["_dispatch"] > marker and not is_flush(r)])
    before = dispatches([r for r in rows if r["_dispatch"] < marker and not is_flush(r) and not is_marker(r)])
    totals, kernels, blits = {}, set(), set()
    for d, name, values in measured:
        for counter, value in values.items():
            totals[counter] = totals.get(counter, 0.0) + value
        (blits if name.startswith("__amd_rocclr_") else kernels).add(d)
    warmups = None
    length = len(measured)
    if length and len(before) >= WARMUPS * length:
        groups = [before[len(before) - (WARMUPS - i) * length:len(before) - (WARMUPS - 1 - i) * length]
                  for i in range(WARMUPS)]
        names = [name for _, name, _ in measured]
        if all([name for _, name, _ in g] == names for g in groups):
            warmups = []
            for g in groups:
                sums = {}
                for _, _, values in g:
                    for counter, value in values.items():
                        sums[counter] = sums.get(counter, 0.0) + value
                warmups.append(sums)
    return totals, len(kernels), len(blits), warmups


def leak_verdict(counter, measured, warmups):
    """(suspected: "yes" / "no" / "not checked", smallest warm-up or None, threshold or None)"""
    if warmups is None or counter not in LEAK_FLOOR or any(counter not in w for w in warmups):
        return "not checked", None, None
    smallest = min(w[counter] for w in warmups)
    threshold = max(LEAK_RELATIVE * smallest, LEAK_FLOOR[counter])
    return ("yes" if measured - smallest > threshold else "no"), smallest, threshold


def gpu_target(directory):
    """The gfx target of the GPU agent, from rocprofv3's agent_info.csv."""
    for f in glob.glob(os.path.join(directory, "**", "*agent_info.csv"), recursive=True):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("Agent_Type") == "GPU" and r.get("Name", "").startswith("gfx"):
                    return r["Name"]
    return None


def flush_kib(log_file):
    """The flush buffer of a rocm_bench --rocprof-target run, in KiB, from its log."""
    with open(log_file, errors="replace") as fh:
        m = FLUSH_RE.search(fh.read())
    if not m:
        raise SystemExit(f"{log_file}: no 'Flushing ... (2 x (L2 ... KiB + MALL ... KiB))' line")
    return 2 * (int(m.group(1)) + int(m.group(2)))


def write_request_bytes(target):
    for family, size in WRITE_REQUEST_BYTES.items():
        if target and target.startswith(family) and target[len(family):len(family) + 1].isalnum():
            return size
    return None


def labels(out):
    """The <impl>_N<n> directories of <out>/raw, as (label, impl, n), ordered by implementation and N."""
    found = []
    for d in glob.glob(os.path.join(out, "raw", "*")):
        m = re.fullmatch(r"(.+)_N(\d+)", os.path.basename(d))
        if os.path.isdir(d) and m:
            found.append((os.path.basename(d), m.group(1), int(m.group(2))))
    return sorted(found, key=lambda x: (x[1], x[2]))


def counter_dirs(out, label):
    d = os.path.join(out, "raw", label)
    return [(os.path.basename(c), c) for c in sorted(glob.glob(os.path.join(d, "*"))) if os.path.isdir(c)]


def leak_suspects(out):
    """[(label, counter dir name, measured, smallest warm-up, threshold)] of the profiles the guard rejects."""
    suspects = []
    for label, _, _ in labels(out):
        for name, path in counter_dirs(out, label):
            totals, _, _, warmups = summarize(path)
            for counter, value in totals.items():
                verdict, smallest, threshold = leak_verdict(counter, value, warmups)
                if verdict == "yes":
                    suspects.append((label, name, value, smallest, threshold))
    return suspects


def check_open(directory, log_file):
    rows = read_rows(directory)
    flushes = [float(r["Counter_Value"]) for r in rows if is_flush(r) and r["Counter_Name"] == "FETCH_SIZE"]
    if not flushes:
        print(f"{directory}: no FETCH_SIZE records of flushCachesKernel")
        return 1
    expected = flush_kib(log_file)
    smallest = min(flushes)
    print(f"Hardware counter check: FETCH_SIZE of the {len(flushes)} cache flush kernels, which read "
          f"{expected} KiB each: min {smallest:.1f} KiB, max {max(flushes):.1f} KiB (GPU {gpu_target(directory)})")
    return 0 if smallest >= OPEN_FRACTION * expected else 3


def notes(sizes):
    return (
        f"{NOTES_BEGIN}\n"
        "Counters (summarize_counters.py): every counter is the sum over the dispatches of the measured execution "
        "(after measureMarkerKernel), without the cache flush kernels, with the HIP runtime's blit kernels.\n"
        "FETCH_SIZE (KiB) is rocprofiler-sdk's derived counter; on gfx12 it is "
        "(GL2C_EA_RDREQ_32B*32 + 64B*64 + 128B*128 + 256B*256) / 1024. With the caches flushed before every GPU "
        "operation, every kernel reads its inputs from beyond L2.\n"
        "WRITE_SIZE (derived) (KiB) = ((GL2C_EA_WRREQ_sum - GL2C_EA_WRREQ_64B_sum) * S + GL2C_EA_WRREQ_64B_sum * 64)"
        f" / 1024 with S = {', '.join(sizes) or 'not needed'} (gfx12: 256 B, measured with a 256 MiB streaming write on "
        "gfx1200: 257.6 B per request; gfx10/gfx11: 32 B).\n"
        "What the gfx12 counters measure: GL2C_EA counts the requests between L2 (GL2C) and the data fabric (EA). "
        "The Infinity Cache (MALL) is behind the fabric, so these are L2 misses and write-backs, of which a part may be "
        "served by the MALL: not DRAM traffic. rocprofiler-sdk 1.4.1 lists no MALL or DRAM counter for gfx1200.\n"
        "Writes are undercounted: GL2C_EA_WRREQ counts the write-backs that happen while a kernel runs. Lines still "
        "dirty in L2 when a kernel ends are written back without being counted in any dispatch (the cache flush "
        "kernels after them record 0 write requests on gfx1200). So every kernel and runtime blit of an execution can "
        "miss up to the L2 size of writes (4 MiB on the RX 9060), and a kernel that writes less than that can show "
        "about 0 (FKL VF+HF at N = 100: 5 requests for a 2.4 MB resize output, 10906 requests, 2.7 MB, for its 4.9 MB "
        "output). A pipeline of several kernels therefore loses more of its writes than a fused one.\n"
        "Flush leak guard: the counts of a flush kernel occasionally land on the dispatches right after it. A profile "
        "whose measured execution exceeds the smallest of its three warm-up executions (same dispatches, caches also "
        f"flushed) by more than max({LEAK_RELATIVE} x that warm-up, 256 KiB of traffic) is collected again "
        "(rocprof_counters.sh); rejected profiles are in raw_rejected/ and traffic_repeats.csv. In traffic.csv, "
        "rejected_profiles counts them per row, and flush_leak_suspected is yes (still suspected after the last "
        "repeat), no, or not checked (a counter without a threshold, or warm-ups that do not match the measured "
        "dispatches).\n"
        f"{NOTES_END}\n")


def write_notes(path, text):
    """Replaces the notes block of an earlier run in <path>, or appends it."""
    old = ""
    if os.path.exists(path):
        with open(path) as fh:
            old = fh.read()
    if NOTES_BEGIN in old and NOTES_END in old:
        start = old.index(NOTES_BEGIN)
        end = old.index(NOTES_END, start) + len(NOTES_END)
        if old[end:end + 1] == "\n":
            end += 1
        old = old[:start] + old[end:]
    if old and not old.endswith("\n"):
        old += "\n"
    with open(path, "w") as fh:
        fh.write(old + ("\n" if old else "") + text)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--outliers":
        for label, counter, *_ in leak_suspects(sys.argv[2]):
            print(label, counter)
        return 0
    if len(sys.argv) == 4 and sys.argv[1] == "--check-open":
        return check_open(sys.argv[2], sys.argv[3])
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = sys.argv[1]

    # Profiles rejected by the guard, kept in <out>/raw_rejected/<label>/<counter dir>.<attempt>
    repeats_rows, repeats = [], {}
    for d in sorted(glob.glob(os.path.join(out, "raw_rejected", "*", "*"))):
        m = re.fullmatch(r"(.+)\.(\d+)", os.path.basename(d))
        lm = re.fullmatch(r"(.+)_N(\d+)", os.path.basename(os.path.dirname(d)))
        if not (os.path.isdir(d) and m and lm):
            continue
        name, attempt = m.group(1), int(m.group(2))
        totals, _, _, warmups = summarize(d)
        for counter, value in sorted(totals.items()):
            repeats[(lm.group(0), counter)] = repeats.get((lm.group(0), counter), 0) + 1
            _, smallest, _ = leak_verdict(counter, value, warmups)
            repeats_rows.append([lm.group(1), lm.group(2), counter, attempt, f"{value:.1f}",
                                 "" if smallest is None else f"{smallest:.1f}"])

    main_rows, targets, still = [], set(), 0
    for label, impl, n in labels(out):
        totals, verdicts, kernels, blits = {}, {}, None, None
        for _, counter_dir in counter_dirs(out, label):
            t, k, b, warmups = summarize(counter_dir)
            targets.add(gpu_target(counter_dir))
            for counter, value in t.items():
                totals[counter] = value
                verdicts[counter] = leak_verdict(counter, value, warmups)
            if kernels is not None and (k, b) != (kernels, blits):
                raise SystemExit(f"{label}: different dispatch counts in the counter runs ({kernels},{blits} vs {k},{b})")
            kernels, blits = k, b
        if "WRITE_SIZE" not in totals and "GL2C_EA_WRREQ_sum" in totals and "GL2C_EA_WRREQ_64B_sum" in totals:
            target = gpu_target(os.path.join(out, "raw", label))
            size = write_request_bytes(target)
            if size is None:
                raise SystemExit(f"{label}: no WRITE_SIZE and no write request size known for {target} "
                                 f"(WRITE_REQUEST_BYTES in {sys.argv[0]})")
            req, req64 = totals["GL2C_EA_WRREQ_sum"], totals["GL2C_EA_WRREQ_64B_sum"]
            totals["WRITE_SIZE (derived)"] = ((req - req64) * size + req64 * 64) / 1024
            sources = [verdicts[c][0] for c in ("GL2C_EA_WRREQ_sum", "GL2C_EA_WRREQ_64B_sum")]
            derived = "yes" if "yes" in sources else "no" if all(s == "no" for s in sources) else "not checked"
            verdicts["WRITE_SIZE (derived)"] = (derived, None, None)
        for counter, value in sorted(totals.items()):
            verdict, smallest, _ = verdicts[counter]
            still += verdict == "yes"
            main_rows.append([impl, n, counter, f"{value:.1f}", "" if smallest is None else f"{smallest:.1f}", kernels,
                              blits, repeats.get((label, counter), 0), verdict])

    with open(os.path.join(out, "traffic.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["implementation", "N", "counter", "value_summed_over_measured_dispatches",
                    "smallest_of_3_warmup_executions", "pipeline_kernels", "runtime_blit_dispatches",
                    "rejected_profiles", "flush_leak_suspected"])
        w.writerows(main_rows)
    with open(os.path.join(out, "traffic_repeats.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["implementation", "N", "counter", "rejected_attempt", "value_summed_over_measured_dispatches",
                    "smallest_of_3_warmup_executions"])
        w.writerows(repeats_rows)
    sizes = sorted({f"{t}: {write_request_bytes(t)} B" for t in targets if t and write_request_bytes(t)})
    write_notes(os.path.join(out, "platform.txt"), notes(sizes))
    print(f"{len(main_rows)} rows written to {out}/traffic.csv ({len(repeats_rows)} rejected profile values, "
          f"{still} values still suspected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
