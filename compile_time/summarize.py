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
# host_compiler groups every gcc row, as asked. Its three phases are also reported on their own: "preprocessing 4"
# (host code, for cudafe++) and "preprocessing 1" (device code, for cicc) run at the same time, and "compiling" is the
# host code's compile
GCC_PHASES = {"gcc_preprocess_host": "gcc (preprocessing 4)", "gcc_preprocess_device": "gcc (preprocessing 1)",
              "gcc_compile": "gcc (compiling)"}


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


def dash(value):
    return "-" if value is None else value


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
    overlap_check = []  # per timed compile: wall - (sum of phases - the shorter of the two gcc preprocessing passes)
    for impl, batch in order:
        rows = sorted(times.get((impl, batch), []), key=lambda r: int(r["rep"]))
        reps = len(rows)
        wall = [float(r["wall_s"]) for r in rows]
        phase_file = out / f"phases_{impl}_{batch}.csv"
        cats = {c: [] for c in CATEGORIES}
        gcc = {g: [] for g in GCC_PHASES}
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
                for g, phase in GCC_PHASES.items():
                    gcc[g].append(compile_phases.get(phase, 0.0))
            for compile_phases, row in zip(timed, rows):
                prep = [compile_phases.get(GCC_PHASES[g], 0.0) for g in ("gcc_preprocess_host", "gcc_preprocess_device")]
                overlap_check.append(float(row["wall_s"]) - (sum(compile_phases.values()) - min(prep)))
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
        log = (keep / "ptxas.log").read_text() if (keep / "ptxas.log").exists() else ""
        if log.count("Compiling entry function") != len(kernels):
            notes.append(f"{impl} {batch}: ptxas compiled {log.count('Compiling entry function')} entry functions, "
                         f"cuobjdump lists {len(kernels)}")
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
            **{f"{g}_median_s": fmt(median(gcc[g])) for g in GCC_PHASES},
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

    # Machine CPU time per round outside the benchmark's nvcc processes: busy CPU time of /proc/stat minus the user +
    # sys time of the round's compiles
    load_rows = []
    if (out / "load.csv").exists():
        with open(out / "load.csv", newline="") as f:
            for row in csv.DictReader(f):
                rep = int(row["rep"])
                busy = (int(row["busy_jiffies_end"]) - int(row["busy_jiffies_start"])) / int(row["clock_ticks_per_s"])
                nvcc = sum(float(r["user_s"]) + float(r["sys_s"]) for k in times for r in times[k] if int(r["rep"]) == rep)
                if rep > 0:
                    load_rows.append({"rep": rep, "machine_busy_cpu_s": f"{busy:.2f}", "nvcc_cpu_s": f"{nvcc:.2f}",
                                      "other_cpu_s": f"{busy - nvcc:.2f}", "loadavg_1min": row["loadavg_1min"]})
        with open(out / "compile_time_load.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rep", "machine_busy_cpu_s", "nvcc_cpu_s", "other_cpu_s",
                                                   "loadavg_1min"])
            writer.writeheader()
            writer.writerows(load_rows)

    print("nvcc --time header: " + ", ".join(header or []))
    if overlap_check:
        print(f"With --threads 0 only gcc (preprocessing 1) and (preprocessing 4) run at the same time: in every timed "
              f"compile, wall = sum of the phases - the shorter of the two, to within "
              f"{max(abs(x) for x in overlap_check):.3f} s. host_compiler groups all the gcc rows as asked; "
              f"gcc compile is the host code's compile alone.")
    if load_rows:
        print(f"CPU time used outside the benchmark's nvcc, per round: max "
              f"{max(float(r['other_cpu_s']) for r in load_rows):.2f} s (compile_time_load.csv)")
    print()
    print("| TU | N | wall median (min-max) s | cudafe++ s | cicc s | ptxas s | gcc compile s | host compiler s "
          "(all gcc) | kernels | registers max | spill stores max | cubin bytes | object bytes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['impl']} | {s['batch']} | {s['wall_median_s']} ({s['wall_min_s']}-{s['wall_max_s']}) | "
              f"{s['cudafe_median_s']} | {s['cicc_median_s']} | {s['ptxas_median_s']} | {s['gcc_compile_median_s']} | "
              f"{s['host_compiler_median_s']} | {s['kernels']} | {dash(s['registers_max'])} | "
              f"{dash(s['spill_stores_bytes_max'])} | {s['cubin_bytes']} | {s['object_bytes']} |")
    for note in notes:
        print("Note: " + note)


if __name__ == "__main__":
    main()
