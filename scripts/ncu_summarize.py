#!/usr/bin/env python3
"""Sums the memory traffic of every pipeline from the Nsight Compute reports.

Run without sudo after the profiling script:
    ncu_summarize.py        after ncu_dram.sh: results/ncu/*.ncu-rep  -> results/dram_traffic.csv
    ncu_summarize.py l2     after ncu_l2.sh:   results/ncu_l2/*.ncu-rep -> results/l2_traffic.csv

For every pipeline, N and cache control it writes the number of kernels and the bytes summed over all of them. Every
profile is one execution on the regions of iteration 0 (scripts/ncu_common.sh). ncu_dram.sh takes every profile more
than once (the DRAM counters are device-wide, so a display on the same GPU adds traffic to some of them): the row of a
pipeline, N and cache control is its profile with the fewest DRAM bytes read, with the number of profiles and the
largest DRAM read among them next to it.
"""
import csv
import os
import pathlib
import re
import shutil
import subprocess
import sys

# Override with the environment variables NCU and BVF_RESULTS; by default results/ is next to this repository
NCU = os.environ.get("NCU") or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
RESULTS = pathlib.Path(os.environ.get("BVF_RESULTS", pathlib.Path(__file__).resolve().parents[2] / "results"))
# <benchmark folder>__<implementation>_N<N>_cache-<all|none>[-<repeat>] (scripts/ncu_common.sh, ncu_dram.sh)
REPORT_NAME = re.compile(r"^(?P<bench>[a-z-]+)__(?P<impl>.+)_N(?P<n>\d+)_cache-(?P<cache>all|none)(?:-(?P<repeat>\d+))?$")
SECTOR_BYTES = 32

MODES = {
    "dram": {
        "dir": "ncu",
        "csv": "dram_traffic.csv",
        "metrics": ["dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum"],
    },
    "l2": {
        "dir": "ncu_l2",
        "csv": "l2_traffic.csv",
        "metrics": ["lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex_op_write.sum",
                    "lts__t_bytes.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum"],
    },
}


def number(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def totals_of(report, metrics):
    out = subprocess.run([NCU, "--import", str(report), "--csv", "--page", "raw", "--print-units", "base",
                          "--metrics", ",".join(metrics)], capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(line for line in out.splitlines() if line.startswith('"')))
    header = rows[0]
    col = {m: header.index(m) for m in metrics}
    kernels = [r for r in rows[1:] if number(r[col[metrics[0]]]) is not None]  # skips the units row
    return {m: sum(number(r[col[m]]) for r in kernels) for m in metrics}, len(kernels)


def row_for(mode, totals):
    mb = 1e6
    if mode == "dram":
        return {
            "dram_read_MB": totals["dram__bytes_read.sum"] / mb,
            "dram_write_MB": totals["dram__bytes_write.sum"] / mb,
            "dram_total_MB": (totals["dram__bytes_read.sum"] + totals["dram__bytes_write.sum"]) / mb,
            "sum_kernel_time_us_under_ncu": totals["gpu__time_duration.sum"] / 1e3,
        }
    read = totals["lts__t_sectors_srcunit_tex_op_read.sum"] * SECTOR_BYTES
    write = totals["lts__t_sectors_srcunit_tex_op_write.sum"] * SECTOR_BYTES
    return {
        "sm_to_l2_read_MB": read / mb,
        "sm_to_l2_write_MB": write / mb,
        "sm_to_l2_total_MB": (read + write) / mb,
        "l2_total_bytes_MB": totals["lts__t_bytes.sum"] / mb,
        "dram_read_MB": totals["dram__bytes_read.sum"] / mb,
        "dram_write_MB": totals["dram__bytes_write.sum"] / mb,
        "sum_kernel_time_us_under_ncu": totals["gpu__time_duration.sum"] / 1e3,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dram"
    cfg = MODES[mode]
    # The benchmarks and pipelines the profiling scripts profile, in the order of the summary
    benches = ["fkl-vs-opencv-cuda", "fkl-vs-npp", "fkl-ablation"]
    order = ["opencv_cuda", "npp", "fkl_vf_hf", "fkl_bvf"]
    summary = []
    for report in sorted((RESULTS / cfg["dir"]).glob("*.ncu-rep")):
        match = REPORT_NAME.match(report.stem)
        if not match:
            continue
        if match["impl"] not in order or match["bench"] not in benches:
            print(f"Skipping {report.name}: not a pipeline the profiling scripts profile (a report of an earlier run?)")
            continue
        totals, kernels = totals_of(report, cfg["metrics"])
        row = {
            "implementation": match["impl"],
            "benchmark": match["bench"],
            "N": int(match["n"]),
            "cache_control": "cold (flush before every kernel)" if match["cache"] == "all" else "warm (no flush)",
            "kernels": kernels,
            **row_for(mode, totals),
        }
        key = (row["benchmark"], row["implementation"], row["N"], row["cache_control"])
        same = next((s for s in summary if (s["benchmark"], s["implementation"], s["N"], s["cache_control"]) == key), None)
        if same is None:
            summary.append({**row, "profiles": 1, "dram_read_max_of_profiles_MB": row["dram_read_MB"]})
            continue
        # Several profiles of the same execution: keep the one with the fewest DRAM bytes read
        profiles, largest = same["profiles"] + 1, max(same["dram_read_max_of_profiles_MB"], row["dram_read_MB"])
        if row["dram_read_MB"] < same["dram_read_MB"]:
            same.update(row)
        same.update({"profiles": profiles, "dram_read_max_of_profiles_MB": largest})
    if not summary:
        sys.exit(f"No reports of the profiled pipelines in {RESULTS / cfg['dir']}: run the profiling script first")
    summary.sort(key=lambda s: (s["cache_control"], s["N"], benches.index(s["benchmark"]),
                                order.index(s["implementation"])))
    with open(RESULTS / cfg["csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for s in summary:
            # MB with 6 decimals: whole bytes, so that small differences (one region is a few hundred bytes) survive
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in s.items()})
    for s in summary:
        values = "  ".join(f"{k} {v:9.3f}" for k, v in s.items() if isinstance(v, float) and "time" not in k)
        print(f"{s['cache_control'][:4]} N={s['N']:3d} {s['benchmark']:18s} {s['implementation']:24s} "
              f"kernels {s['kernels']:3d}  {values}")
    print(f"Written {RESULTS / cfg['csv']}")


if __name__ == "__main__":
    main()
