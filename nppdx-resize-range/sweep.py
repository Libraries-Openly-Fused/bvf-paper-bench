#!/usr/bin/env python3
"""The NPPDx resize-range sweep: which Resize<J, K, Method> ratios NPPDx v0.1.2 accepts inside a fused kernel.

    nppdx-resize-range/sweep.py <output directory> [--jobs N] [--max-jk M] [--nppdx <checkout>]

Steps, all into the output directory:
1. Builds probe/resize_rules.cu and runs it over every coprime J:K up to --max-jk (default 128), every method, SM 890
   and 900 and every tile NPPDx accepts: rules/rules_by_ratio.csv (the best tile of every ratio at every level) and
   rules/rules_summary.csv (the largest factor of every method at every level). These are NPPDx's own constexpr
   functions, evaluated at run time.
2. Lists the points to compile (points.txt): the ratios of RATIOS x the four methods x SM 890 and 900 x the tiles of
   TILES; for every ratio of RATIOS and every level, the best tile of rules_by_ratio.csv; every maximum of
   rules_summary.csv, and the same tile with the next integer factor.
3. Evaluates the rules at every point (predicted.csv).
4. Compiles every point with probe/resize_probe.cu (nvcc, -arch=sm_89 for SM 890 and sm_90 for SM 900), in parallel.
   A point NPPDx rejects fails to compile: its static_assert texts are recorded. A point that compiles is run: it
   prints its tiles, shared memory and block. On the RTX 4090 every SM 890 point is also launched once with
   fused_resize.cu's block (--launch) and once with 32 threads (--launch 32), and compared with the CPU reference.
5. Writes sweep.csv (one line per point) and agreement.txt (the rules against the compiler and the program).

No timing: this is capability data, one launch per point.
"""

import argparse
import concurrent.futures
import csv
import os
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
METHODS = ["nearest", "bilinear", "bicubic", "lanczos3"]
SMS = [890, 900]
ARCH = {890: "sm_89", 900: "sm_90"}

# J:K, J input samples to K output samples
RATIOS = (
    [(1, 1)]
    + [(1, k) for k in list(range(2, 17)) + [20, 24, 29, 30, 31, 32, 33, 48, 49]]
    + [(j, 1) for j in list(range(2, 17)) + [20, 24, 32, 40, 41]]
    + [(2, 3), (3, 2), (3, 4), (4, 3), (5, 7), (7, 5), (2, 5), (5, 2), (3, 5), (5, 3), (4, 5), (5, 4),
       (5, 16), (16, 5), (3, 8), (8, 3), (9, 16), (16, 9), (15, 16), (16, 15), (4, 9), (9, 4), (135, 28)]
)
# Input tiles, multiples of NPPDx's 4x2 cell: from the smallest to the largest NPPDx accepts on SM 890 (x * y * 12 bytes
# within 99 KB), and larger ones for SM 900 (227 KB)
TILES_890 = [(4, 2), (4, 4), (8, 4), (8, 8), (12, 6), (16, 8), (16, 16), (20, 10), (24, 12), (24, 24), (28, 14),
             (32, 16), (32, 32), (40, 20), (48, 24), (48, 48), (60, 30), (64, 32), (64, 64), (96, 48), (128, 64),
             (96, 88), (132, 64), (4224, 2)]
TILES_900 = TILES_890 + [(128, 128), (136, 136), (192, 96), (256, 64), (9680, 2)]

LEVELS = ["compiles", "compiles_j_divides", "fits_smem", "fits_smem_j_divides", "fits_smem_example_block",
          "fits_smem_example_block_j_divides"]


def nvcc_command(nppdx, sm, method, j, k, tx, ty, out):
    return ["nvcc", "-std=c++20", "-O3", f"-arch={ARCH[sm]}", "--expt-relaxed-constexpr", "-rdc=false",
            "-ftemplate-depth=1000", f"-I{nppdx}/include", f"-I{nppdx}/example/nppdx/common",
            f"-DPROBE_J={j}", f"-DPROBE_K={k}", f"-DPROBE_METHOD={method}", f"-DPROBE_TX={tx}", f"-DPROBE_TY={ty}",
            f"-DPROBE_SM={sm}", str(HERE / "probe" / "resize_probe.cu"), "-o", str(out)]


def parse_kv(line, prefix):
    """'prefix a=1,b=2' -> {'a': '1', 'b': '2'}"""
    body = line[len(prefix):].strip()
    out = {}
    for part in body.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


STATIC_ASSERT = re.compile(r'error: static assertion failed with "(.*?)"')
ERROR = re.compile(r"error: (.*)")


def run_point(point, nppdx, bindir, launch):
    sm, method, j, k, tx, ty = point
    name = f"sm{sm}_{method}_{j}_{k}_{tx}x{ty}"
    binary = bindir / name
    result = {"sm": sm, "method": method, "J": j, "K": k, "tile": f"{tx}x{ty}"}
    start = time.time()
    try:
        proc = subprocess.run(nvcc_command(nppdx, sm, method, j, k, tx, ty, binary), capture_output=True, text=True,
                              timeout=1800)
        result["compile_seconds"] = f"{time.time() - start:.1f}"
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        result["compile_seconds"] = f"{time.time() - start:.1f}"
        result["compiled"] = "timeout"
        return result
    if proc.returncode != 0:
        result["compiled"] = "no"
        asserts = []
        for m in STATIC_ASSERT.finditer(log):
            if m.group(1) not in asserts:
                asserts.append(m.group(1))
        result["static_asserts"] = " | ".join(asserts)
        if not asserts:
            errors = [m.group(1) for m in ERROR.finditer(log)]
            result["other_errors"] = " | ".join(dict.fromkeys(errors))[:500]
        return result
    result["compiled"] = "yes"
    try:
        info = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120).stdout
        for line in info.splitlines():
            if line.startswith("point "):
                result.update({"probe_" + key: v for key, v in parse_kv(line, "point").items()})
        if launch and sm == 890:
            for tag, args in (("launch", ["--launch"]), ("launch32", ["--launch", "32"])):
                out = subprocess.run([str(binary)] + args, capture_output=True, text=True, timeout=600).stdout
                for line in out.splitlines():
                    if line.startswith("device "):
                        result.update({"device_" + key: v for key, v in parse_kv(line, "device").items()})
                    if line.startswith("launch "):
                        result.update({f"{tag}_{key}": v for key, v in parse_kv(line, "launch").items()})
    finally:
        binary.unlink(missing_ok=True)
    return result


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--max-jk", type=int, default=128)
    parser.add_argument("--nppdx", type=Path, default=REPO / "nppdx")
    parser.add_argument("--no-launch", action="store_true", help="do not launch the SM 890 points")
    args = parser.parse_args()

    out = args.output.resolve()
    (out / "rules").mkdir(parents=True, exist_ok=True)
    bindir = out / "bin"
    bindir.mkdir(exist_ok=True)
    nppdx = args.nppdx.resolve()

    def capture(cmd, cwd=None):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()
        except OSError as e:
            return str(e)

    with open(out / "environment.txt", "w") as f:
        f.write(f"bvf-paper-bench: {capture(['git', 'rev-parse', 'HEAD'], cwd=REPO)} "
                f"({capture(['git', 'status', '--porcelain', '--', 'nppdx-resize-range'], cwd=REPO) or 'clean'})\n")
        f.write(f"nppdx: {capture(['git', 'describe', '--always', '--tags'], cwd=nppdx)} "
                f"{capture(['git', 'rev-parse', 'HEAD'], cwd=nppdx)}\n")
        f.write(f"nvcc: {capture(['nvcc', '--version']).splitlines()[-1]}\n")
        f.write(f"gpu: {capture(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])}\n")
        f.write(f"max J and K of the rules: {args.max_jk}\n")
        f.write(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. NPPDx's rules over every ratio and tile
    rules = bindir / "resize_rules"
    subprocess.run(["nvcc", "-std=c++20", "-O3", "-arch=sm_89", "--expt-relaxed-constexpr", f"-I{nppdx}/include",
                    str(HERE / "probe" / "resize_rules.cu"), "-o", str(rules)], check=True)
    with open(out / "rules" / "rules_stdout.txt", "w") as f:
        subprocess.run([str(rules), "summary", str(args.max_jk), str(out / "rules")], check=True, stdout=f)

    # 2. The points
    points = {}

    def add(sm, method, j, k, tx, ty, why):
        key = (sm, method, j, k, tx, ty)
        points.setdefault(key, []).append(why)

    for sm in SMS:
        tiles = TILES_890 if sm == 890 else TILES_900
        for method in METHODS:
            for j, k in RATIOS:
                for tx, ty in tiles:
                    add(sm, method, j, k, tx, ty, "grid")
    ratios = set(RATIOS)
    for row in read_csv(out / "rules" / "rules_by_ratio.csv"):
        j, k = int(row["J"]), int(row["K"])
        if (j, k) not in ratios:
            continue
        for level in LEVELS:
            tile = row[f"{level}_best_tile"]
            if tile:
                tx, ty = map(int, tile.split("x"))
                add(int(row["sm"]), row["method"], j, k, tx, ty, f"best:{level}")
    for row in read_csv(out / "rules" / "rules_summary.csv"):
        sm, method, j, k = int(row["sm"]), row["method"], int(row["J"]), int(row["K"])
        tx, ty = map(int, row["tile"].split("x"))
        add(sm, method, j, k, tx, ty, f"max:{row['direction']}:{row['level']}")
        # The next integer factor with the same tile
        if row["direction"] == "upscale" and j == 1:
            add(sm, method, 1, k + 1, tx, ty, f"next:{row['direction']}:{row['level']}")
        if row["direction"] == "downscale" and k == 1:
            add(sm, method, j + 1, 1, tx, ty, f"next:{row['direction']}:{row['level']}")
    keys = sorted(points)
    with open(out / "points.txt", "w") as f:
        for key in keys:
            f.write(" ".join(map(str, key)) + "\n")

    # 3. The rules at every point
    subprocess.run([str(rules), "points", str(out / "points.txt"), str(out / "predicted.csv")], check=True)
    predicted = {}
    for row in read_csv(out / "predicted.csv"):
        tx, ty = map(int, row["tile"].split("x"))
        predicted[(int(row["sm"]), row["method"], int(row["J"]), int(row["K"]), tx, ty)] = row

    # 4. Compile and run every point, the slowest first (big tiles, big ratios)
    order = sorted(keys, key=lambda p: -(p[4] * p[5] * max(p[2], p[3])))
    results = {}
    print(f"{len(order)} points, {args.jobs} jobs", flush=True)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_point, p, nppdx, bindir, not args.no_launch): p for p in order}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if i % 200 == 0 or i == len(order):
                print(f"  {i}/{len(order)} after {time.time() - start:.0f} s", flush=True)

    # 5. sweep.csv and the agreement
    columns = ["sm", "method", "J", "K", "factor", "direction", "tile", "why", "compiled", "compile_seconds",
               "static_asserts", "other_errors", "predicted_compiles", "predicted_failing_checks", "agrees",
               "box_halo", "resize_halo", "input_storage", "blurred_storage", "intermediate_storage", "output_storage",
               "smem_bytes", "shared_memory_limit", "fits_smem", "block", "example_block_fits", "j_divides_tile",
               "probe_matches_rules", "device_max_shared_memory_per_block_optin",
               "launch_result", "launch_error", "launch_block", "launch_image", "launch_output", "launch_max_diff",
               "launch_max_diff_interior", "launch_interior_values_over_1",
               "launch32_result", "launch32_error", "launch32_max_diff", "launch32_max_diff_interior",
               "launch32_interior_values_over_1"]
    disagreements, mismatches, rows = [], [], []
    for key in keys:
        r, p = results[key], predicted[key]
        sm, method, j, k, tx, ty = key
        f = Fraction(k, j)
        row = {"sm": sm, "method": method, "J": j, "K": k, "tile": f"{tx}x{ty}", "why": " ".join(points[key]),
               "factor": f"{float(f if f >= 1 else 1 / f):.4f}",
               "direction": "upscale" if k > j else ("downscale" if j > k else "identity")}
        for c in ("compiled", "compile_seconds", "static_asserts", "other_errors"):
            row[c] = r.get(c, "")
        row["predicted_compiles"] = p["predicted_compiles"]
        row["predicted_failing_checks"] = p["failing_checks"]
        row["agrees"] = "yes" if (r.get("compiled") == "yes") == (p["predicted_compiles"] == "yes") else "no"
        if row["agrees"] == "no":
            disagreements.append(row)
        for c in ("box_halo", "resize_halo", "input_storage", "blurred_storage", "intermediate_storage",
                  "output_storage", "smem_bytes", "shared_memory_limit", "fits_smem", "block", "example_block_fits",
                  "j_divides_tile"):
            row[c] = p[c]
        if r.get("compiled") == "yes":
            same = all(r.get("probe_" + c) == p[c] for c in ("box_halo", "resize_halo", "input_storage",
                                                            "blurred_storage", "intermediate_storage",
                                                            "output_storage", "smem_bytes", "block"))
            row["probe_matches_rules"] = "yes" if same else "no"
            if not same:
                mismatches.append((row, r))
        row["device_max_shared_memory_per_block_optin"] = r.get("device_max_shared_memory_per_block_optin", "")
        for tag in ("launch", "launch32"):
            row[f"{tag}_result"] = r.get(f"{tag}_result", "")
            row[f"{tag}_error"] = r.get(f"{tag}_error", "")
            row[f"{tag}_max_diff"] = r.get(f"{tag}_max_diff", "")
            row[f"{tag}_max_diff_interior"] = r.get(f"{tag}_max_diff_interior", "")
            row[f"{tag}_interior_values_over_1"] = r.get(f"{tag}_interior_values_over_1", "")
        row["launch_block"] = r.get("launch_block", "")
        row["launch_image"] = r.get("launch_image", "")
        row["launch_output"] = r.get("launch_output", "")
        rows.append(row)
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "agreement.txt", "w") as f:
        compiled = sum(1 for r in rows if r["compiled"] == "yes")
        f.write(f"points: {len(rows)}, compiled: {compiled}, rejected: {len(rows) - compiled}\n")
        f.write(f"rules vs compiler, accepted or not: {len(rows) - len(disagreements)} agree, "
                f"{len(disagreements)} disagree\n")
        f.write(f"rules vs probe, tiles / shared memory / block of the compiled points: "
                f"{compiled - len(mismatches)} match, {len(mismatches)} differ\n")
        for row in disagreements:
            f.write(f"DISAGREE {row['sm']} {row['method']} {row['J']}:{row['K']} {row['tile']} compiled="
                    f"{row['compiled']} predicted={row['predicted_compiles']} ({row['predicted_failing_checks']}) "
                    f"{row['static_asserts']} {row['other_errors']}\n")
        for row, r in mismatches:
            f.write(f"MISMATCH {row['sm']} {row['method']} {row['J']}:{row['K']} {row['tile']} "
                    f"{ {k: v for k, v in r.items() if k.startswith('probe_')} }\n")
    print((out / "agreement.txt").read_text())
    return 0 if not disagreements and not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
