#!/usr/bin/env python3
"""Lines of code of the FKL and NPPDx versions of each pipeline the paper compares (README, "Lines of code").

    nppdx-wins/loc/count_loc.py [--repo <checkout>] [--manifest <manifest.json>] [--show]

Every item of the manifest is a list of line ranges of a file at a commit of this repository. The count of an item is
the number of lines in its ranges that are neither blank nor a comment (a line whose first non-blank characters are //,
or that lies inside a /* */ block, tracked over the whole file). The files are read with git show <commit>:<path>, so
the counts do not depend on the checked-out branch, only on the commits being present: 3845b60 (main), b7f6e88
(nppdx-npp-border-align), 96cc71d (exp3-neighborhood) and 76a9136 (nppdx-wins). The manifest's "rule" says which code
the ranges cover.

--repo: the checkout to read the commits from (default: the one this script is in). --manifest: default manifest.json
next to this script. --show: print every range with a + on each counted line. The script compares every total with the
manifest's "expected" (the number in the paper) and exits with 1 if any differs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def file_lines(repo, commit, path):
    out = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"git show {commit}:{path} failed in {repo} (fetch the branches that hold the commits):\n{out.stderr}")
    return out.stdout.split("\n")


def classify(lines):
    """True for every line that counts (neither blank nor a comment), tracking /* */ blocks over the whole file."""
    counts = []
    in_block = False
    for line in lines:
        s = line.strip()
        if in_block:
            counts.append(False)
            if "*/" in s:
                in_block = False
            continue
        if s == "" or s.startswith("//"):
            counts.append(False)
        elif s.startswith("/*"):
            counts.append(False)
            if "*/" not in s:
                in_block = True
        else:
            counts.append(True)
    return counts


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=None, help="checkout to read the commits from (default: this script's)")
    parser.add_argument("--manifest", default=str(here / "manifest.json"))
    parser.add_argument("--show", action="store_true", help="print every range with a + on each counted line")
    args = parser.parse_args()
    repo = args.repo or subprocess.run(["git", "-C", str(here), "rev-parse", "--show-toplevel"], check=True,
                                       capture_output=True, text=True).stdout.strip()
    with open(args.manifest) as f:
        manifest = json.load(f)
    cache = {}
    results = []
    for item in manifest["items"]:
        total = 0
        parts = []
        for rng in item["ranges"]:
            key = (rng["commit"], rng["path"])
            if key not in cache:
                lines = file_lines(repo, *key)
                cache[key] = (lines, classify(lines))
            lines, counts = cache[key]
            a, b = rng["lines"]
            n = sum(counts[a - 1:b])
            total += n
            parts.append((rng["path"], a, b, n, rng.get("what", "")))
            if args.show:
                print(f"--- {item['name']}: {rng['path']} {a}-{b} ({rng.get('what', '')}): {n} lines")
                for i in range(a, b + 1):
                    mark = "+" if counts[i - 1] else " "
                    print(f"{mark}{i:5d}  {lines[i - 1]}")
        results.append((item["name"], total, item.get("expected"), parts))
    if args.show:
        print()
    print(f"{'item':48s} lines  expected")
    mismatches = 0
    for name, total, expected, parts in results:
        check = "" if expected is None else ("ok" if total == expected else f"MISMATCH ({expected})")
        mismatches += expected is not None and total != expected
        print(f"{name:48s} {total:5d}  {check}")
        for path, a, b, n, what in parts:
            print(f"    {n:4d}  {path}:{a}-{b}  {what}")
    print(f"\n{len(results) - mismatches} of {len(results)} totals as expected")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
