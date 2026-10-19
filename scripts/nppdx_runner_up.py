#!/usr/bin/env python3
"""The runner-up NPPDx configuration of every batch size, to re-time it against FKL with --configs.

    nppdx_runner_up.py <benchmark>_nppdx_block_sweep_GPU_NVIDIA.csv <output configuration CSV>

Reads the configuration sweep of a run of benchmark_nppdx or benchmark_nppdx_npp (every configuration validated and
its NPPDx kernel timed alone) and writes, in the format of <benchmark>_nppdx_config_GPU_NVIDIA.csv that --configs
reads, the second fastest configuration within its tolerance for every batch size: the fastest one that the run did
not choose. Running the benchmark with --configs <output> then compares it with FKL with the same protocol and regions as
the chosen one, which shows how much the choice matters.
"""
import csv
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    rows = list(csv.DictReader(open(sys.argv[1]), skipinitialspace=True))
    tolerance = "Within tolerance"
    if rows and tolerance not in rows[0]:
        sys.exit(f"{sys.argv[1]} has no '{tolerance}' column: it comes from an older version of the benchmark")
    runner_up = {}
    for r in rows:
        batch = int(r["Batch size"])
        if r[tolerance] != "yes" or r["Chosen"] == "yes":
            continue
        if batch not in runner_up or float(r["MedianTime"]) < float(runner_up[batch]["MedianTime"]):
            runner_up[batch] = r
    batches = sorted({int(r["Batch size"]) for r in rows})
    missing = [n for n in batches if n not in runner_up]
    if missing:
        sys.exit(f"No second configuration within tolerance for N = {missing}")
    with open(sys.argv[2], "w") as f:
        f.write("Batch size, Tile, Requested BlockDim (0 = NPPDx recommended), Actual BlockSize, Blocks per crop, "
                "Dynamic shared memory bytes\n")
        for n in batches:
            r = runner_up[n]
            f.write(f"{n}, {r['Tile']}, {r['Requested BlockDim (0 = NPPDx recommended)']}, {r['Actual BlockSize']}, "
                    f"{r['Blocks per crop']}, {r['Dynamic shared memory bytes']}\n")
    for n in batches:
        r = runner_up[n]
        print(f"N={n:3d}: runner-up tile {r['Tile']} block {r['Requested BlockDim (0 = NPPDx recommended)']} "
              f"(kernel median {float(r['MedianTime']) * 1000:.2f} us)")


if __name__ == "__main__":
    main()
