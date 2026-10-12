#!/usr/bin/env python3
"""Compulsory-read model of every Figure 1 pipeline, next to the Nsight Compute measurements, for any workload.

Run without sudo after ncu_dram.sh and ncu_l2.sh and the two ncu_summarize.py modes:
    ncu_traffic_model.py        BVF_RESULTS (default results/ next to this repository) -> traffic_model.csv in it

The workload (frame, crop, resize and border sizes, batch sizes, seed and margin of the crops) is read from the
Workload line of the NPPDx benchmark's platform file in BVF_RESULTS/nppdx, and the crop positions are generated as
common/bench_geometry.h's makeCrops() does. Without measurements (dram_traffic.csv, l2_traffic.csv) only the model
columns that do not need them are written.

Model, per execution:
  - a full-frame conversion reads the whole NV12 frame;
  - per region, the crop stage reads the 8 bit RGB region, the resize stage the float3 region and the border stage the
    float3 resized region, once each;
  - OpenCV's and NPP's three pointwise passes read the whole output three times;
  - NPPDx + NPP (nppdx_npp): its NPPDx kernel reads the NV12 source footprint of the regions (below), NPP's border
    (nppiSet of the 8 bit RGB output) reads nothing, its 8u -> 32f conversion reads the 8 bit RGB output and its three
    pointwise passes the float3 output three times; the per-launch terms of its six kernels are not modelled
    (implied_fixed_term_KB is the measurement minus the model);
  - FKL BVF and NPPDx read a fixed per-launch term (code, constants, parameters), measured as the DRAM reads after a
    flush at the smallest N (N = 1) minus the footprint of its regions, plus the NV12 source footprint of the regions
    in 32 byte sectors: their luma rows and chroma rows, each sector counted once even where regions overlap (a cold
    execution reads it from DRAM once). Both resizes only read inside the region (FKL samples
    at dst * scale and NPPDx at pixel centers, both clamped to the region). The NPPDx benchmarks may choose a different
    kernel configuration (tile, block) for every N (<impl>/<benchmark>_nppdx_config_GPU_NVIDIA.csv), with its own
    fixed term: the NPPDx model is only given for the N whose configuration is that of the smallest N. For every N,
    implied_fixed_term_KB is the measurement minus the footprint, to see how constant the fixed term is.
  - Reads between the SMs and L2 (lts__t_sectors_srcunit_tex_op_read) only see data loads, not the fixed term, so for
    FKL BVF and NPPDx they are compared with the compulsory part alone, the region footprint (sm_to_l2_compulsory_MB;
    for NPPDx + NPP plus the reads of its NPP kernels):
    the sum over the regions, without removing overlaps, since each region is loaded by its own thread blocks. It is
    not a model of the requests: measured_sm_to_l2_read_over_compulsory above 1 is the re-requested part. FKL BVF's
    bilinear taps request some sectors more than once. NPPDx splits a region into nppdx_blocks_per_crop tiles, and
    every block loads its tile plus the resize halo, so neighboring tiles request the sectors of their shared edges
    again; with one block per crop it requests each sector of the region once.
Measured DRAM reads: both scripts profile every pipeline with the caches flushed before every kernel, and both collect
dram__bytes_read.sum. The DRAM counters are device-wide (a GPU that also drives a display adds traffic of other
processes), so the smaller of the two measurements is used, and both are kept. The summaries must have byte precision
(ncu_summarize.py writes MB with 6 decimals): a region is a few hundred bytes.
"""
import csv
import os
import pathlib
import re
import sys

RESULTS = pathlib.Path(os.environ.get("BVF_RESULTS", pathlib.Path(__file__).resolve().parents[2] / "results"))
SECTOR = 32
PIPELINES = [("paper-bench", "opencv_cuda"), ("paper-bench", "npp"), ("paper-bench", "fkl_vf_hf"),
             ("paper-bench", "fkl_bvf"), ("FKL-vs-NPPDx", "nppdx_npp"), ("FKL-vs-NPPDx", "nppdx"),
             ("FKL-vs-NPPDx", "fkl_bvf")]
# The benchmark program of each NPPDx implementation; its results are in BVF_RESULTS/<implementation>
NPPDX_BENCHMARKS = {"nppdx": "benchmark_figure1_bvf", "nppdx_npp": "benchmark_figure1_nppdx_npp"}


class MT19937:
    """std::mt19937: its output sequence is fixed by the C++ standard."""

    def __init__(self, seed):
        self.mt = [0] * 624
        self.index = 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF

    def __call__(self):
        if self.index >= 624:
            for i in range(624):
                y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            self.index = 0
        y = self.mt[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return y ^ (y >> 18)


def uniform_int(gen, lo, hi):
    """bench::geometry::uniformInt: Lemire's multiply and shift with rejection."""
    rng = (hi - lo + 1) & 0xFFFFFFFF
    product = gen() * rng
    low = product & 0xFFFFFFFF
    if low < rng:
        threshold = ((1 << 32) - rng) % rng
        while low < threshold:
            product = gen() * rng
            low = product & 0xFFFFFFFF
    return lo + (product >> 32)


def workload():
    """The workload of BVF_RESULTS, from the Workload line of the NPPDx benchmark's platform file."""
    platform = RESULTS / "nppdx" / "benchmark_figure1_bvf_platform.txt"
    line = next((l for l in open(platform) if l.startswith("Workload: ")), None)
    if line is None:
        sys.exit(f"No Workload line in {platform}")
    patterns = {
        "frame": r"(\d+)x(\d+) NV12 frame",
        "batches": r"N in \{([\d, ]+)\}",
        "crop": r"crops of (\d+)x(\d+) at uniform random even positions at least (\d+) pixels from the frame edges "
                r"\(seed (\d+)",
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
        "crop_w": int(m["crop"][1]), "crop_h": int(m["crop"][2]), "margin": int(m["crop"][3]),
        "crop_seed": int(m["crop"][4]),
        "res_w": int(m["resize"][1]), "res_h": int(m["resize"][2]),
    }
    w["out_w"], w["out_h"] = w["res_w"], w["res_h"] + int(m["border"][1]) + int(m["border"][2])
    return w


def make_crops(w):
    """bench::geometry::makeCrops(): even origins at least margin pixels from the frame edges."""
    gen = MT19937(w["crop_seed"])
    crops = []
    for _ in range(max(w["batches"])):
        x = 2 * uniform_int(gen, w["margin"] // 2, (w["frame_w"] - w["crop_w"] - w["margin"]) // 2)
        y = 2 * uniform_int(gen, w["margin"] // 2, (w["frame_h"] - w["crop_h"] - w["margin"]) // 2)
        crops.append((x, y))
    return crops


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return {(r["benchmark"], r["implementation"], int(r["N"]), r["cache_control"][:4]): r for r in csv.DictReader(open(path))}


def nppdx_configs(impl):
    """The NPPDx configuration (tile, requested block) and blocks per crop of every N, if the NPPDx run wrote them."""
    path = RESULTS / impl / f"{NPPDX_BENCHMARKS[impl]}_nppdx_config_GPU_NVIDIA.csv"
    if not path.exists():
        return {}
    rows = list(csv.reader(open(path), skipinitialspace=True))[1:]
    return {int(r[0]): (f"{r[1]} block {r[2]}", r[4]) for r in rows if r and r[0].isdigit()}


def main():
    w = workload()
    crops = make_crops(w)
    frame = w["frame_w"] * w["frame_h"] * 3 // 2
    region8 = w["crop_w"] * w["crop_h"] * 3
    region32 = w["crop_w"] * w["crop_h"] * 12
    resized32 = w["res_w"] * w["res_h"] * 12
    output8 = w["out_w"] * w["out_h"] * 3
    output = w["out_w"] * w["out_h"] * 12

    def footprint(x):  # NV12 bytes of one region, in whole 32 byte sectors (the frame pitch is a multiple of 32)
        sectors_per_row = (x + w["crop_w"] - 1) // SECTOR - x // SECTOR + 1
        return (w["crop_h"] + w["crop_h"] // 2) * sectors_per_row * SECTOR

    def requested(n):  # the footprints of the first n regions, summed: what their loads request from L2
        return sum(footprint(x) for x, _ in crops[:n])

    def sources(n):  # the NV12 sectors of the first n regions, each once: what a cold execution reads from DRAM
        sectors = set()
        for x, y in crops[:n]:
            columns = range(x // SECTOR, (x + w["crop_w"] - 1) // SECTOR + 1)
            sectors.update(("Y", row, c) for row in range(y, y + w["crop_h"]) for c in columns)
            sectors.update(("UV", row, c) for row in range(y // 2, y // 2 + w["crop_h"] // 2) for c in columns)
        return len(sectors) * SECTOR

    dram, l2 = load("dram_traffic.csv"), load("l2_traffic.csv")
    configs = {impl: nppdx_configs(impl) for impl in NPPDX_BENCHMARKS}
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
        single = impl in ("fkl_bvf", "nppdx")
        sources_read = single or impl == "nppdx_npp"  # reads the NV12 source footprint of the regions
        config = lambda n, impl=impl: configs.get(impl, {}).get(n, ("", ""))[0]
        blocks = lambda n, impl=impl: configs.get(impl, {}).get(n, ("", ""))[1]
        fixed, anchor = None, None
        if single:
            measured_ns = [n for n in w["batches"] if dram_cold(bench, impl, n) is not None]
            if measured_ns:
                anchor = min(measured_ns)
                fixed = dram_cold(bench, impl, anchor) - sources(anchor)
                one_pass = [n for n in measured_ns if None in (measured(dram, bench, impl, n, "cold", "dram_read_MB"),
                                                                 measured(l2, bench, impl, n, "cold", "dram_read_MB"))]
                if one_pass:
                    warnings.append(f"{bench} {impl}: only one cold measurement at N = {one_pass}, so an outlier from "
                                    f"other GPU traffic cannot be discarded")
                same = [dram_cold(bench, impl, n) - sources(n) for n in measured_ns if config(n) == config(anchor)]
                median = sorted(same)[len(same) // 2]
                if abs(median - fixed) > 1000:
                    warnings.append(f"{bench} {impl}: the fixed term at N = {anchor} ({fixed / 1e3:.3f} KB) differs "
                                    f"from the median over N ({median / 1e3:.3f} KB) by more than 1 KB")
        for n in w["batches"]:
            note = ""
            if impl in ("opencv_cuda", "npp"):
                model = frame + n * (region8 + region32 + resized32) + 3 * n * output
            elif impl == "fkl_vf_hf":
                model = frame + n * (region8 + region32 + resized32)
            elif impl == "nppdx_npp":
                model = sources(n) + n * output8 + 3 * n * output
            elif fixed is None:
                model = None
            elif config(n) != config(anchor):
                model = None
                note = f"NPPDx configuration {config(n)} differs from {config(anchor)} at N = {anchor}"
            else:
                model = fixed + sources(n)
            if impl == "nppdx_npp":
                sm_l2_compulsory = requested(n) + n * output8 + 3 * n * output
            else:
                sm_l2_compulsory = requested(n) if single else model
            cold = dram_cold(bench, impl, n)
            sm_l2 = measured(l2, bench, impl, n, "cold", "sm_to_l2_read_MB")
            ratio = lambda v, m: f"{v / m:.4f}" if v is not None and m else ""
            mb = lambda v: f"{v / 1e6:.6f}" if v is not None else ""
            rows.append({
                "benchmark": bench, "implementation": impl, "N": n, "nppdx_configuration": config(n),
                "nppdx_blocks_per_crop": blocks(n),
                "model_compulsory_read_MB": mb(model),
                "region_source_footprint_MB": mb(sources(n)) if sources_read else "",
                "fixed_term_KB": f"{fixed / 1e3:.3f}" if fixed is not None and model is not None else "",
                "implied_fixed_term_KB": (f"{(cold - (sources(n) if single else model)) / 1e3:.3f}"
                                          if sources_read and cold is not None else ""),
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
                "note": note,
            })
    with open(RESULTS / "traffic_model.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Workload: {w['crop_w']}x{w['crop_h']} crops resized to {w['res_w']}x{w['res_h']}, output "
          f"{w['out_w']}x{w['out_h']}, N in {w['batches']}"
          + ("" if dram or l2 else "; no measurements yet, so no FKL BVF or NPPDx model (it needs a measurement)"))
    for r in rows:
        print(f"{r['benchmark']:12s} {r['implementation']:24s} N={r['N']:3d}  model {r['model_compulsory_read_MB'] or '-':>9} MB"
              f"  DRAM cold {r['measured_dram_read_cold_MB'] or '-':>9} ({r['measured_dram_read_cold_over_model'] or '-'})"
              f"  SM->L2 {r['measured_sm_to_l2_read_MB'] or '-':>9} ({r['measured_sm_to_l2_read_over_compulsory'] or '-'})"
              + (f"  implied fixed {r['implied_fixed_term_KB']} KB" if r['implied_fixed_term_KB'] else "")
              + (f"  [{r['note']}]" if r['note'] else ""))
    for warning in warnings:
        print("WARNING:", warning)
    print(f"Written {RESULTS / 'traffic_model.csv'}")


if __name__ == "__main__":
    main()
