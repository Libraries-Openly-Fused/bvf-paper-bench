#!/usr/bin/env python3
"""Summarizes rocm/scripts/rocprof_counters.sh. For every <impl>_N<n> directory of <dir>/raw
(one subdirectory per counter, one rocprofv3 run each), the counters of the dispatches after measureMarkerKernel (the
measured execution), without the cache flush kernels, summed per counter. The HIP runtime's own blit kernels
(__amd_rocclr_*, which execute memsets and some copies) are included in the sums, since their memory traffic is part of
the pipeline's, and counted separately from the pipeline's kernels. Writes <dir>/traffic.csv; for gfx12, which has no WRITE_SIZE in rocprofiler-sdk 1.4.1, WRITE_SIZE (KiB) is
derived from GL2C_EA_WRREQ_sum and GL2C_EA_WRREQ_64B_sum with WRITE_SIZE's gfx10/gfx11 formula."""
import csv
import glob
import os
import re
import sys


def summarize(directory):
    """Sums of every counter over the measured dispatches of one rocprofv3 output directory."""
    files = glob.glob(os.path.join(directory, "**", "*counter_collection.csv"), recursive=True)
    if not files:
        raise SystemExit(f"no counter_collection.csv under {directory}")
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    for r in rows:
        r["_dispatch"] = int(r["Dispatch_Id"])
    marker = sorted({r["_dispatch"] for r in rows if "measureMarkerKernel" in r["Kernel_Name"]})
    if len(marker) != 1:
        raise SystemExit(f"{directory}: expected one measureMarkerKernel dispatch, found {marker}")
    measured = [r for r in rows if r["_dispatch"] > marker[0] and "flushCachesKernel" not in r["Kernel_Name"]]
    totals, kernels, blits = {}, set(), set()
    for r in measured:
        totals[r["Counter_Name"]] = totals.get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        (blits if r["Kernel_Name"].startswith("__amd_rocclr_") else kernels).add(r["_dispatch"])
    return totals, len(kernels), len(blits)


def main():
    out = sys.argv[1]
    main_rows = []
    for d in sorted(glob.glob(os.path.join(out, "raw", "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = re.fullmatch(r"(.+)_N(\d+)", name)
        if not m:
            continue
        totals, kernels, blits = {}, None, None
        for counter_dir in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(counter_dir):
                continue
            t, k, b = summarize(counter_dir)
            totals.update(t)
            if kernels is not None and (k, b) != (kernels, blits):
                raise SystemExit(f"{d}: different dispatch counts in the counter runs ({kernels},{blits} vs {k},{b})")
            kernels, blits = k, b
        if "WRITE_SIZE" not in totals and "GL2C_EA_WRREQ_sum" in totals and "GL2C_EA_WRREQ_64B_sum" in totals:
            req, req64 = totals["GL2C_EA_WRREQ_sum"], totals["GL2C_EA_WRREQ_64B_sum"]
            totals["WRITE_SIZE (derived)"] = ((req - req64) * 32 + req64 * 64) / 1024
        for counter, value in sorted(totals.items()):
            main_rows.append([m.group(1), m.group(2), counter, f"{value:.1f}", kernels, blits])
    header = ["counter", "value_summed_over_measured_dispatches", "pipeline_kernels", "runtime_blit_dispatches"]
    with open(os.path.join(out, "traffic.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["implementation", "N"] + header)
        w.writerows(main_rows)
    print(f"{len(main_rows)} rows written to {out}/traffic.csv")


if __name__ == "__main__":
    main()
