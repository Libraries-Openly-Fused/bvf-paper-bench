#!/usr/bin/env python3
"""Compulsory-read model of the profiled pipelines (OpenCV CUDA, NPP, FKL VF+HF, FKL BVF), next to the Nsight Compute
measurements.

Run without sudo after ncu_dram.sh and ncu_l2.sh and the two ncu_summarize.py modes:
    ncu_traffic_model.py        BVF_RESULTS (default results/ next to this repository) -> traffic_model.csv in it

Every profile is one execution on the regions of iteration 0 of common/bench_geometry.h (batch N: the first N), which
the profiling scripts write to ncu_regions.csv in BVF_RESULTS; their sizes are 20x10, 32x16 and 48x24 by turns, so the
bytes of a region depend on its size. The frame, resize and border sizes and the batch sizes are read from the Workload
line of the platform file of FKL's ablation in BVF_RESULTS/fkl-ablation. Without measurements (dram_traffic.csv,
l2_traffic.csv) only the model columns that do not need them are written.

Model, per execution, summed over the N regions, each with its own width w and height h:
  - a full-frame conversion reads the whole NV12 frame;
  - OpenCV and NPP: per region the crop stage reads the 8 bit RGB region (3 w h bytes), the resize stage the float3
    region (12 w h) and the border stage the float3 resized region, once each; the three pointwise passes read the
    whole output three times;
  - FKL VF+HF: its crop kernel is one FKL batch, which runs the threads of the largest of the N regions in every plane,
    so it reads that box (48x24 as soon as the N regions include one of that size) of 8 bit RGB for every region; then
    the resize reads the float3 region (12 w h) and the last kernel the float3 resized region;
  - FKL BVF reads a fixed per-launch term (code, constants, parameters), measured as the DRAM reads after a flush at the
    smallest N (N = 1) minus the footprint of its region, plus the NV12 source footprint of the regions in 32 byte
    sectors: their luma rows and chroma rows, each sector counted once even where regions overlap (a cold execution
    reads it from DRAM once). Its resize only reads inside the region (it samples at dst * scale, clamped to the
    region). For every N, implied_fixed_term_KB is the measurement minus the footprint, to see how constant the fixed
    term is.
  - Reads between the SMs and L2 (lts__t_sectors_srcunit_tex_op_read) only see data loads, not the fixed term, so for
    FKL BVF they are compared with the compulsory part alone, the region footprint (sm_to_l2_compulsory_MB): the sum
    over the regions, without removing overlaps, since each region is loaded by its own thread blocks. It is not a
    model of the requests: measured_sm_to_l2_read_over_compulsory above 1 is the re-requested part (FKL BVF's bilinear
    taps request some sectors more than once).
Measured DRAM reads: both scripts profile every pipeline with the caches flushed before every kernel (the L2 script at
the batch sizes of its warm profiles), and both collect dram__bytes_read.sum. The DRAM counters are device-wide (a GPU
that also drives a display adds traffic of other processes), so where both measured, the smaller of the two is used, and
both are kept. The summaries must have byte precision (ncu_summarize.py writes MB with 6 decimals): a region is a few
hundred bytes.
"""
import csv
import os
import pathlib
import re
import sys

RESULTS = pathlib.Path(os.environ.get("BVF_RESULTS", pathlib.Path(__file__).resolve().parents[2] / "results"))
SECTOR = 32
# (benchmark folder, implementation) of every profiled pipeline (scripts/ncu_common.sh)
PIPELINES = [("fkl-vs-opencv-cuda", "opencv_cuda"), ("fkl-vs-npp", "npp"), ("fkl-ablation", "fkl_vf_hf"),
             ("fkl-ablation", "fkl_bvf")]


def workload():
    """The workload of BVF_RESULTS, from the Workload line of the platform file of FKL's ablation."""
    platform = RESULTS / "fkl-ablation" / "platform.txt"
    line = next((l for l in open(platform) if l.startswith("Workload: ")), None)
    if line is None:
        sys.exit(f"No Workload line in {platform}")
    patterns = {
        "frame": r"(\d+)x(\d+) NV12 frame",
        "batches": r"N in \{([\d, ]+)\}",
        "resize": r"bilinear resize to (\d+)x(\d+)",
        "border": r"(\d+) black rows added on top and (\d+) at the bottom",
    }
    m = {k: re.search(p, line) for k, p in patterns.items()}
    missing = [k for k, v in m.items() if v is None]
    if missing:
        sys.exit(f"Cannot read {', '.join(missing)} from the Workload line of {platform}")
    w = {
        "frame_w": int(m["frame"][1]), "frame_h": int(m["frame"][2]),
        "batches": [int(n) for n in m["batches"][1].split(",")],
        "res_w": int(m["resize"][1]), "res_h": int(m["resize"][2]),
    }
    w["out_w"], w["out_h"] = w["res_w"], w["res_h"] + int(m["border"][1]) + int(m["border"][2])
    return w


def regions():
    """The regions of iteration 0, in order, as (x, y, w, h): ncu_regions.csv, written by the profiling scripts."""
    path = RESULTS / "ncu_regions.csv"
    if not path.exists():
        sys.exit(f"No {path}: the profiling scripts write it (from the benchmarks' --regions 0)")
    rows = list(csv.DictReader(open(path)))
    if not rows or any(int(r["iteration"]) != 0 for r in rows):
        sys.exit(f"{path} does not hold the regions of iteration 0")
    return [(int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])) for r in rows]


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return {(r["benchmark"], r["implementation"], int(r["N"]), r["cache_control"][:4]): r for r in csv.DictReader(open(path))}


def main():
    w = workload()
    regs = regions()
    frame = w["frame_w"] * w["frame_h"] * 3 // 2
    resized32 = w["res_w"] * w["res_h"] * 12
    output = w["out_w"] * w["out_h"] * 12

    def footprint(x, rw, rh):  # NV12 bytes of one region, in whole 32 byte sectors (the frame pitch is a multiple of 32)
        sectors_per_row = (x + rw - 1) // SECTOR - x // SECTOR + 1
        return (rh + rh // 2) * sectors_per_row * SECTOR

    def requested(n):  # the footprints of the first n regions, summed: what their loads request from L2
        return sum(footprint(x, rw, rh) for x, _, rw, rh in regs[:n])

    def sources(n):  # the NV12 sectors of the first n regions, each once: what a cold execution reads from DRAM
        sectors = set()
        for x, y, rw, rh in regs[:n]:
            columns = range(x // SECTOR, (x + rw - 1) // SECTOR + 1)
            sectors.update(("Y", row, c) for row in range(y, y + rh) for c in columns)
            sectors.update(("UV", row, c) for row in range(y // 2, y // 2 + rh // 2) for c in columns)
        return len(sectors) * SECTOR

    def area(n):  # pixels of the first n regions
        return sum(rw * rh for _, _, rw, rh in regs[:n])

    def box(n):  # pixels of the threads of FKL VF+HF's crop kernel: the largest of the first n regions, in every plane
        return n * max(rw for _, _, rw, _ in regs[:n]) * max(rh for _, _, _, rh in regs[:n])

    dram, l2 = load("dram_traffic.csv"), load("l2_traffic.csv")
    warnings = []

    def measured(table, bench, impl, n, cache, column):
        row = table.get((bench, impl, n, cache)) if table else None
        return float(row[column]) * 1e6 if row else None

    def dram_cold(bench, impl, n):
        values = [v for v in (measured(dram, bench, impl, n, "cold", "dram_read_MB"),
                              measured(l2, bench, impl, n, "cold", "dram_read_MB")) if v is not None]
        return min(values) if values else None

    rows = []
    for bench, impl in PIPELINES:
        bvf = impl == "fkl_bvf"
        fixed = None
        if bvf:
            measured_ns = [n for n in w["batches"] if dram_cold(bench, impl, n) is not None]
            if measured_ns:
                anchor = min(measured_ns)
                fixed = dram_cold(bench, impl, anchor) - sources(anchor)
                implied = [dram_cold(bench, impl, n) - sources(n) for n in measured_ns]
                median = sorted(implied)[len(implied) // 2]
                if abs(median - fixed) > 1000:
                    warnings.append(f"{bench} {impl}: the fixed term at N = {anchor} ({fixed / 1e3:.3f} KB) differs "
                                    f"from the median over N ({median / 1e3:.3f} KB) by more than 1 KB")
        for n in w["batches"]:
            if impl in ("opencv_cuda", "npp"):
                model = frame + 3 * area(n) + 12 * area(n) + n * resized32 + 3 * n * output
            elif impl == "fkl_vf_hf":
                model = frame + 3 * box(n) + 12 * area(n) + n * resized32
            else:
                model = fixed + sources(n) if fixed is not None else None
            sm_l2_compulsory = requested(n) if bvf else model
            cold = dram_cold(bench, impl, n)
            sm_l2 = measured(l2, bench, impl, n, "cold", "sm_to_l2_read_MB")
            ratio = lambda v, m: f"{v / m:.4f}" if v is not None and m else ""
            mb = lambda v: f"{v / 1e6:.6f}" if v is not None else ""
            sizes = [f"{rw}x{rh}" for _, _, rw, rh in regs[:n]]
            rows.append({
                "benchmark": bench, "implementation": impl, "N": n,
                "regions_20x10": sizes.count("20x10"), "regions_32x16": sizes.count("32x16"),
                "regions_48x24": sizes.count("48x24"), "region_pixels": area(n),
                "model_compulsory_read_MB": mb(model),
                "region_source_footprint_MB": mb(sources(n)) if bvf else "",
                "fixed_term_KB": f"{fixed / 1e3:.3f}" if fixed is not None else "",
                "implied_fixed_term_KB": f"{(cold - sources(n)) / 1e3:.3f}" if bvf and cold is not None else "",
                "measured_dram_read_cold_MB": mb(cold),
                "measured_dram_read_cold_over_model": ratio(cold, model),
                "sm_to_l2_compulsory_MB": mb(sm_l2_compulsory),
                "measured_sm_to_l2_read_MB": mb(sm_l2),
                "measured_sm_to_l2_read_over_compulsory": ratio(sm_l2, sm_l2_compulsory),
                "dram_read_cold_dram_pass_MB": mb(measured(dram, bench, impl, n, "cold", "dram_read_MB")),
                "dram_read_cold_l2_pass_MB": mb(measured(l2, bench, impl, n, "cold", "dram_read_MB")),
                "measured_dram_read_warm_MB": mb(measured(dram, bench, impl, n, "warm", "dram_read_MB")),
                "measured_dram_write_cold_MB": mb(measured(dram, bench, impl, n, "cold", "dram_write_MB")),
                "measured_sm_to_l2_write_MB": mb(measured(l2, bench, impl, n, "cold", "sm_to_l2_write_MB")),
            })
    with open(RESULTS / "traffic_model.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Workload: the regions of iteration 0 (ncu_regions.csv) resized to {w['res_w']}x{w['res_h']}, output "
          f"{w['out_w']}x{w['out_h']}, N in {w['batches']}"
          + ("" if dram or l2 else "; no measurements yet, so no FKL BVF model (it needs a measurement)"))
    for r in rows:
        print(f"{r['benchmark']:18s} {r['implementation']:12s} N={r['N']:3d}  model {r['model_compulsory_read_MB'] or '-':>9} MB"
              f"  DRAM cold {r['measured_dram_read_cold_MB'] or '-':>9} ({r['measured_dram_read_cold_over_model'] or '-'})"
              f"  SM->L2 {r['measured_sm_to_l2_read_MB'] or '-':>9} ({r['measured_sm_to_l2_read_over_compulsory'] or '-'})"
              + (f"  implied fixed {r['implied_fixed_term_KB']} KB" if r['implied_fixed_term_KB'] else ""))
    for warning in warnings:
        print("WARNING:", warning)
    print(f"Written {RESULTS / 'traffic_model.csv'}")


if __name__ == "__main__":
    main()
