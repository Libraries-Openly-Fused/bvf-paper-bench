#!/usr/bin/env python3
"""Sums the memory traffic of every pipeline from the Nsight Compute reports.

Run without sudo after the profiling script:
    ncu_summarize.py        after ncu_dram.sh: results/ncu/*.ncu-rep  -> results/dram_traffic.csv
    ncu_summarize.py l2     after ncu_l2.sh:   results/ncu_l2/*.ncu-rep -> results/l2_traffic.csv

For every pipeline, N and cache control it writes the number of kernels and the bytes summed over all of them.
"""
import csv
import pathlib
import re
import subprocess
import sys

NCU = "/usr/local/cuda-13.3/bin/ncu"
RESULTS = pathlib.Path("/home/oscar/bvf-bench/results")
REPORT_NAME = re.compile(r"^(?P<repo>nppdxrepo_)?(?P<impl>.+)_N(?P<n>\d+)_cache-(?P<cache>all|none)$")
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
    summary = []
    for report in sorted((RESULTS / cfg["dir"]).glob("*.ncu-rep")):
        match = REPORT_NAME.match(report.stem)
        if not match:
            continue
        totals, kernels = totals_of(report, cfg["metrics"])
        summary.append({
            "implementation": match["impl"],
            "benchmark": "FKL-vs-NPPDx" if match["repo"] else "paper-bench",
            "N": int(match["n"]),
            "cache_control": "cold (flush before every kernel)" if match["cache"] == "all" else "warm (no flush)",
            "kernels": kernels,
            **row_for(mode, totals),
        })
    order = ["opencv_cuda", "npp", "nppdx", "fkl_vf_hf_float3_frame", "fkl_vf_hf", "fkl_bvf"]
    summary.sort(key=lambda s: (s["cache_control"], s["N"], s["benchmark"], order.index(s["implementation"])))
    with open(RESULTS / cfg["csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for s in summary:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()})
    for s in summary:
        values = "  ".join(f"{k} {v:9.3f}" for k, v in s.items() if isinstance(v, float) and "time" not in k)
        print(f"{s['cache_control'][:4]} N={s['N']:3d} {s['benchmark']:12s} {s['implementation']:24s} "
              f"kernels {s['kernels']:3d}  {values}")
    print(f"Written {RESULTS / cfg['csv']}")


if __name__ == "__main__":
    main()
