#!/usr/bin/env python3
"""Every zoom ratio that NPPDx resizes correctly on the RTX 4090, per method, for a figure.

    nppdx-resize-range/correct_ratios.py <work directory> <output csv> [--max-jk M] [--jobs N] [--nppdx <checkout>]

The pipeline of sweep.py (ingest -> BoxBlur<5, 5> -> Resize<J, K, Method> -> exgest), SM 890:
1. NPPDx's rules (probe/resize_rules.cu, which agree with the compiler on every point of the sweep) over every coprime
   J:K up to --max-jk (default 512) and every tile: the ratios that have a tile that compiles, fits the RTX 4090's
   shared memory (101376 bytes) with fused_resize.cu's block and that J divides. For each one, the tile with the least
   shared memory.
2. Compiles every ratio at that tile with probe/resize_probe.cu and launches it once (sweep.py's run_point).

One line per ratio: method, J, K, factor (K / J: above 1 an upscale, below 1 a downscale), best_tile, smem_bytes,
launched (yes / no) and correct (yes / no / untested).
- Correct: the launch ran and the interior of its output is within 1 level of the CPU reference, as in README.md.
- No: it ran and differs. This is NPPDx's nearest defect at the center phase (README.md), which does not depend on the
  tile.
The comment lines (#) at the top give the search. The raw launch output is in <work directory>/results.jsonl.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from sweep import HERE, REPO, read_csv, run_point

SM = 890
METHODS = ["nearest", "bilinear", "bicubic", "lanczos3"]
LEVEL = "fits_smem_example_block_j_divides"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-jk", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--nppdx", type=Path, default=REPO / "nppdx")
    args = parser.parse_args()
    work = args.work.resolve()
    bindir = work / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    nppdx = args.nppdx.resolve()

    # 1. NPPDx's rules: the ratios with a correct configuration and their best tile
    rules = bindir / "resize_rules"
    subprocess.run(["nvcc", "-std=c++20", "-O3", "-arch=sm_89", "--expt-relaxed-constexpr", f"-I{nppdx}/include",
                    str(HERE / "probe" / "resize_rules.cu"), "-o", str(rules)], check=True)
    (work / "rules").mkdir(exist_ok=True)
    with open(work / "rules" / "rules_stdout.txt", "w") as f:
        subprocess.run([str(rules), "summary", str(args.max_jk), str(work / "rules")], check=True, stdout=f)
    selected = []
    with open(work / "rules" / "rules_by_ratio.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["sm"]) == SM and r[f"{LEVEL}_best_tile"]:
                selected.append(r)
    print(f"{len(selected)} ratios with a correct configuration by NPPDx's rules", flush=True)

    # 2. Compile and launch every ratio at its best tile
    points = [(SM, r["method"], int(r["J"]), int(r["K"]), *map(int, r[f"{LEVEL}_best_tile"].split("x")))
              for r in selected]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool, \
            open(work / "results.jsonl", "w") as raw:
        futures = {pool.submit(run_point, p, nppdx, bindir, True): p for p in points}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
            raw.write(json.dumps(results[futures[fut]]) + "\n")

    rows, problems = [], []
    for r, p in zip(selected, points):
        res = results[p]
        smem = r[f"{LEVEL}_best_smem"]
        if res.get("compiled") != "yes" or res.get("probe_smem_bytes") != smem:
            problems.append(f"{p}: compiled={res.get('compiled')}, smem {res.get('probe_smem_bytes')} vs rules {smem}")
        launched = res.get("launch_result", "") != ""
        if res.get("launch_result") == "ran" and int(res["launch_interior_pixels"]) > 0:
            correct = "yes" if int(res["launch_max_diff_interior"]) <= 1 else "no"
        else:
            correct = "no" if launched else "untested"
        j, k = p[2], p[3]
        rows.append({"method": r["method"], "J": j, "K": k, "factor": f"{k / j:.6g}", "best_tile": f"{p[4]}x{p[5]}",
                     "smem_bytes": smem, "launched": "yes" if launched else "no", "correct": correct,
                     "interior_max_diff": res.get("launch_max_diff_interior", "")})
    rows.sort(key=lambda r: (METHODS.index(r["method"]), r["K"] / r["J"], r["J"]))

    largest_j = max(r["J"] for r in rows)
    largest_k = max(r["K"] for r in rows)
    with open(args.output, "w", newline="") as f:
        f.write(f"# NPPDx v0.1.2 Resize<J, K, Method> after BoxBlur<5, 5> in one kernel (fused_resize.cu's pipeline), "
                f"RTX 4090 (SM 890, 101376 bytes of shared memory)\n")
        f.write(f"# Every coprime J:K with J, K <= {args.max_jk} that has a correct configuration (a tile that "
                f"compiles, fits with the example's block and that J divides), by NPPDx's rules; the largest J is "
                f"{largest_j} and the largest K {largest_k}, so the list is complete for J, K <= 32\n")
        f.write("# best_tile: the tile with the least shared memory; launched / correct: that tile compiled and "
                "launched once, correct = the interior within 1 level of the CPU reference (no = NPPDx's nearest "
                "defect at the center phase, README.md)\n")
        w = csv.DictWriter(f, fieldnames=["method", "J", "K", "factor", "best_tile", "smem_bytes", "launched",
                                          "correct"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        ok = [r for r in sub if r["correct"] == "yes"]
        bad = [f"{r['J']}:{r['K']} ({r['interior_max_diff']})" for r in sub if r["correct"] != "yes"]
        factors = [r["K"] / r["J"] for r in ok]
        print(f"{method:8s} {len(sub):3d} by the rules, {len(ok):3d} correct, x{min(factors):.4g} to "
              f"x{max(factors):.4g}; not correct: {', '.join(bad) or 'none'}")
    for problem in problems:
        print("PROBLEM", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
