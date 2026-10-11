#!/usr/bin/env python3
"""Summary of a compile-time run of compile_time/run.sh.

    compile_time/summarize.py <results directory>

Reads compile_time.csv (the warm-up compiles, rep 0, are left out), the phases_<IMPL>_<BATCH>.csv files written by
nvcc --time and the keep_<IMPL>_<BATCH> folders of the untimed -Xptxas -v --keep compiles, and writes
compile_time_summary.csv (one row per TU), compile_time_kernels.csv (one row per kernel) and compile_time_phases.csv
(the median of every phase nvcc reported, by phase name), and prints the summary as a Markdown table.
"""

import csv
import re
import statistics
import sys
from pathlib import Path

UNIT_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
CATEGORIES = ("cudafe", "cicc", "ptxas", "fatbinary", "host_compiler", "other")


def category(phase):
    name = phase.lower()
    if name.startswith("cudafe"):
        return "cudafe"
    if name.startswith("cicc"):
        return "cicc"
    if name.startswith("ptxas"):
        return "ptxas"
    if name.startswith("fatbinary"):
        return "fatbinary"
    if name.startswith(("gcc", "g++", "cl", "clang")):
        return "host_compiler"
    return "other"


def read_phases(path):
    """The compiles of one nvcc --time file, in order: a list of {phase name: seconds} (repeated phases summed)."""
    with open(path, newline="") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f) if row]
    header, compiles, current = rows[0], [], {}
    col = {name: i for i, name in enumerate(header)}
    for row in rows[1:]:
        if row == header:
            continue
        phase = row[col["phase name"]]
        seconds = float(row[col["metric"]]) * UNIT_SECONDS[row[col["unit"]]]
        current[phase] = current.get(phase, 0.0) + seconds
        if phase.startswith("nvcc (driver)"):  # the last row of every compile
            compiles.append(current)
            current = {}
    if current:
        compiles.append(current)
    return header, compiles


def kernel_facts(keep):
    """Per kernel: registers and stack frame (cuobjdump --dump-resource-usage), spills (ptxas -v)."""
    kernels, name = {}, None
    usage = (keep / "resource_usage.txt").read_text() if (keep / "resource_usage.txt").exists() else ""
    for line in usage.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            kernels[name] = {"registers": None, "stack": None, "spill_stores": None, "spill_loads": None}
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+)", line)
        if m and name:
            kernels[name]["registers"], kernels[name]["stack"] = int(m.group(1)), int(m.group(2))
    log = (keep / "ptxas.log").read_text() if (keep / "ptxas.log").exists() else ""
    for m in re.finditer(r"Function properties for (\S+)\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, "
                         r"(\d+) bytes spill loads", log):
        if m.group(1) in kernels:
            kernels[m.group(1)].update(spill_stores=int(m.group(3)), spill_loads=int(m.group(4)))
            if kernels[m.group(1)]["stack"] is None:
                kernels[m.group(1)]["stack"] = int(m.group(2))
    return kernels


def sizes(keep):
    ptx = cubin = 0
    if (keep / "sizes.txt").exists():
        for line in (keep / "sizes.txt").read_text().splitlines():
            size, name = line.split(maxsplit=1)
            if name.endswith(".ptx"):
                ptx += int(size)
            elif name.endswith(".cubin"):
                cubin += int(size)
    obj = int((keep / "object_bytes.txt").read_text()) if (keep / "object_bytes.txt").exists() else None
    return ptx, cubin, obj


def median(values):
    return statistics.median(values) if values else None


def fmt(value, digits=3):
    return "" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def main():
    out = Path(sys.argv[1])
    times, order = {}, []
    with open(out / "compile_time.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["impl"], int(row["batch"]))
            if key not in order:
                order.append(key)
            if int(row["rep"]) == 0:
                continue
            times.setdefault(key, []).append(row)

    summary, kernel_rows, phase_rows, header, notes = [], [], [], None, []
    for impl, batch in order:
        rows = times.get((impl, batch), [])
        reps = len(rows)
        wall = [float(r["wall_s"]) for r in rows]
        phase_file = out / f"phases_{impl}_{batch}.csv"
        cats = {c: [] for c in CATEGORIES}
        phase_sum = []
        if phase_file.exists():
            header, compiles = read_phases(phase_file)
            if len(compiles) != reps + 1:
                notes.append(f"{impl} {batch}: {len(compiles)} compiles in {phase_file.name}, expected {reps + 1}")
            warmup, timed = compiles[0], compiles[1:]
            for compile_phases in timed:
                totals = dict.fromkeys(CATEGORIES, 0.0)
                for phase, seconds in compile_phases.items():
                    totals[category(phase)] += seconds
                for c in CATEGORIES:
                    cats[c].append(totals[c])
                phase_sum.append(sum(totals.values()))
            for phase in sorted({p for c in timed for p in c}):
                phase_rows.append({"impl": impl, "batch": batch, "phase": phase, "category": category(phase),
                                   "median_s": fmt(median([c.get(phase, 0.0) for c in timed]), 4)})
            if wall and timed:
                warm_wall = next(float(r["wall_s"]) for r in csv.DictReader(open(out / "compile_time.csv"))
                                 if (r["impl"], int(r["batch"]), int(r["rep"])) == (impl, batch, 0))
                if abs(warm_wall - median(wall)) > 0.2 * median(wall):
                    notes.append(f"{impl} {batch}: warm-up compile took {warm_wall:.2f} s, timed median "
                                 f"{median(wall):.2f} s")
        keep = out / f"keep_{impl}_{batch}"
        kernels = kernel_facts(keep)
        for name, k in kernels.items():
            kernel_rows.append({"impl": impl, "batch": batch, "kernel": name, "registers": k["registers"],
                                "stack_frame_bytes": k["stack"], "spill_stores_bytes": k["spill_stores"],
                                "spill_loads_bytes": k["spill_loads"]})
        ptx, cubin, obj = sizes(keep)

        def kmax(field):
            values = [k[field] for k in kernels.values() if k[field] is not None]
            return max(values) if values else None

        summary.append({
            "impl": impl, "batch": batch, "reps": reps,
            "wall_median_s": fmt(median(wall)), "wall_min_s": fmt(min(wall) if wall else None),
            "wall_max_s": fmt(max(wall) if wall else None),
            "user_median_s": fmt(median([float(r["user_s"]) for r in rows])),
            "sys_median_s": fmt(median([float(r["sys_s"]) for r in rows])),
            "maxrss_median_mb": fmt(median([int(r["maxrss_kb"]) / 1024 for r in rows]), 1),
            **{f"{c}_median_s": fmt(median(cats[c])) for c in CATEGORIES},
            "kernels": len(kernels), "registers_max": kmax("registers"),
            "spill_stores_bytes_max": kmax("spill_stores"), "stack_frame_bytes_max": kmax("stack"),
            "ptx_bytes": ptx, "cubin_bytes": cubin, "object_bytes": obj,
            "phases_sum_median_s": fmt(median(phase_sum)),
        })

    with open(out / "compile_time_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(out / "compile_time_kernels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "batch", "kernel", "registers", "stack_frame_bytes",
                                               "spill_stores_bytes", "spill_loads_bytes"])
        writer.writeheader()
        writer.writerows(kernel_rows)
    with open(out / "compile_time_phases.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "batch", "phase", "category", "median_s"])
        writer.writeheader()
        writer.writerows(phase_rows)

    print("nvcc --time header: " + ", ".join(header or []))
    print("Phases run in parallel with --threads 0, so their sum (phases_sum) can exceed the wall time.\n")
    print("| TU | N | wall median (min-max) s | cudafe++ s | cicc s | ptxas s | host compiler s | phases sum s | "
          "kernels | registers max | spill stores max | cubin bytes | object bytes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['impl']} | {s['batch']} | {s['wall_median_s']} ({s['wall_min_s']}-{s['wall_max_s']}) | "
              f"{s['cudafe_median_s']} | {s['cicc_median_s']} | {s['ptxas_median_s']} | {s['host_compiler_median_s']} | "
              f"{s['phases_sum_median_s']} | {s['kernels']} | {s['registers_max']} | {s['spill_stores_bytes_max']} | "
              f"{s['cubin_bytes']} | {s['object_bytes']} |")
    for note in notes:
        print("Note: " + note)


if __name__ == "__main__":
    main()
