#!/usr/bin/env python3
"""Sums the DRAM traffic of every pipeline from the Nsight Compute reports written by ncu_dram.sh.

Run without sudo after ncu_dram.sh: reads results/ncu/*.ncu-rep through `ncu --import` and writes
results/dram_traffic.csv with, per pipeline, N and cache control, the number of kernels and the DRAM bytes read and
written summed over all of them.
"""
import csv
import pathlib
import re
import subprocess

NCU = "/usr/local/cuda-13.3/bin/ncu"
RESULTS = pathlib.Path("/home/oscar/bvf-bench/results")
METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum"]
REPORT_NAME = re.compile(r"^(?P<repo>nppdxrepo_)?(?P<impl>.+)_N(?P<n>\d+)_cache-(?P<cache>all|none)$")


def number(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def summarize(report):
    out = subprocess.run([NCU, "--import", str(report), "--csv", "--page", "raw", "--print-units", "base",
                          "--metrics", ",".join(METRICS)], capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(line for line in out.splitlines() if line.startswith('"')))
    header = rows[0]
    col = {m: header.index(m) for m in METRICS}
    kernels = [r for r in rows[1:] if number(r[col[METRICS[0]]]) is not None]  # skips the units row
    return {m: sum(number(r[col[m]]) for r in kernels) for m in METRICS}, len(kernels)


def main():
    summary = []
    for report in sorted((RESULTS / "ncu").glob("*.ncu-rep")):
        match = REPORT_NAME.match(report.stem)
        if not match:
            continue
        totals, kernels = summarize(report)
        summary.append({
            "implementation": match["impl"],
            "benchmark": "FKL-vs-NPPDx" if match["repo"] else "paper-bench",
            "N": int(match["n"]),
            "cache_control": "cold (flush before every kernel)" if match["cache"] == "all" else "warm (no flush)",
            "kernels": kernels,
            "dram_read_MB": totals["dram__bytes_read.sum"] / 1e6,
            "dram_write_MB": totals["dram__bytes_write.sum"] / 1e6,
            "dram_total_MB": (totals["dram__bytes_read.sum"] + totals["dram__bytes_write.sum"]) / 1e6,
            "sum_kernel_time_us_under_ncu": totals["gpu__time_duration.sum"] / 1e3,
        })
    order = ["opencv_cuda", "npp", "nppdx", "fkl_vf_hf_float3_frame", "fkl_vf_hf", "fkl_bvf"]
    summary.sort(key=lambda s: (s["cache_control"], s["N"], s["benchmark"], order.index(s["implementation"])))
    with open(RESULTS / "dram_traffic.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for s in summary:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()})
    for s in summary:
        print(f"{s['cache_control'][:4]} N={s['N']:3d} {s['benchmark']:12s} {s['implementation']:24s} "
              f"kernels {s['kernels']:3d}  read {s['dram_read_MB']:9.3f} MB  write {s['dram_write_MB']:9.3f} MB  "
              f"total {s['dram_total_MB']:9.3f} MB")
    print(f"Written {RESULTS / 'dram_traffic.csv'}")


if __name__ == "__main__":
    main()
