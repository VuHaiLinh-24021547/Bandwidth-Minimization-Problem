#!/usr/bin/env python3
"""
benchmark_all.py
================
Runs solvers on instances from the input directory and records results in CSV files,
aligned with main.py standards:
  - Computes theoretical graph-theoretic lower bounds (LB: deg, diam, nbhd)
  - Computes initial bandwidth before permutation
  - Uses the unified 11-column CSV schema matching main.py
  - Segregates outputs into category folders:
      output/small_instances/
      output/large_instances/

Columns in CSV:
  instance, n, edges, initial_bw, lb, solver, status, bandwidth, encoding_time, solve_time, total_time

Options:
  --input-dir    Directory containing instance files (default: input)
  --output-dir   Directory where category folders will be saved (default: output)
  --category     Run only 'small_instances', 'large_instances', or 'all' (default: all)
  --time-limit   Time limit per solver execution in seconds (default: 60)
  --solvers      Specific solvers to run (default: cplex_cp cplex_mip ortools gurobi)
  --instances    Optional list of specific instance names or patterns to run
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from hb_parser import load_hb_file
from utils import (
    compute_caprara_salazar_lb,
    compute_original_bandwidth,
    compute_theoretical_lb,
    verify_solution,
)


def get_solver_registry() -> Dict[str, Tuple[str, any]]:
    """Return map of solver key -> (Display name, solve_function)."""
    registry = {}
    try:
        from solver.CP import CplexCP
        registry["cplex_cp"] = ("CPLEX CP", CplexCP.solve)
    except Exception:
        registry["cplex_cp"] = ("CPLEX CP", None)

    try:
        from solver.CP import CPSat
        registry["ortools"] = ("OR-Tools CP-SAT", CPSat.solve)
    except Exception:
        registry["ortools"] = ("OR-Tools CP-SAT", None)

    try:
        from solver.ILP import CplexMIP
        registry["cplex_mip"] = ("CPLEX MIP", CplexMIP.solve)
    except Exception:
        registry["cplex_mip"] = ("CPLEX MIP", None)

    try:
        from solver.ILP import gurobi
        registry["gurobi"] = ("Gurobi MIP", gurobi.solve)
    except Exception:
        registry["gurobi"] = ("Gurobi MIP", None)

    return registry


def get_instance_category(filepath: str) -> str:
    """Determine whether an instance is small or large based on its directory path."""
    norm = filepath.replace("\\", "/").lower()
    if "large_instances" in norm or "large instances" in norm:
        return "large_instances"
    elif "small_instances" in norm or "small instances" in norm:
        return "small_instances"
    else:
        parent = os.path.basename(os.path.dirname(filepath))
        return parent if parent not in ("", ".", "input") else "small_instances"


def get_instance_stem(filepath: str) -> str:
    """Extract instance name without .mtx.rnd (or other extensions)."""
    filename = os.path.basename(filepath)
    if filename.endswith(".mtx.rnd"):
        return filename[:-8]
    elif filename.endswith(".rnd"):
        return filename[:-4]
    elif filename.endswith(".mtx"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def find_instance_files(input_dir: str, category_filter: str = "all") -> Dict[str, List[str]]:
    """Find all .mtx.rnd (and fallback data) files grouped by category."""
    patterns = [
        os.path.join(input_dir, "**", "*.mtx.rnd"),
        os.path.join(input_dir, "**", "*.mtx"),
    ]
    all_files_set = set()
    for pat in patterns:
        all_files_set.update(glob.glob(pat, recursive=True))

    grouped: Dict[str, List[str]] = {
        "small_instances": [],
        "large_instances": [],
    }

    for f in all_files_set:
        cat = get_instance_category(f)
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(f)

    for cat in grouped:
        grouped[cat].sort(key=lambda p: os.path.basename(p))

    if category_filter != "all":
        return {k: v for k, v in grouped.items() if k == category_filter}

    return grouped


def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Run solvers on instances and export results aligned with main.py."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default="input",
        help="Input folder path containing instance files (default: input)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output folder path where category folders will be stored (default: output)",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["all", "small_instances", "large_instances"],
        default="all",
        help="Category of instances to run (default: all)",
    )
    parser.add_argument(
        "--time-limit", "-t",
        type=int,
        default=120,
        help="Timeout in seconds for each solver (default: 60)",
    )
    parser.add_argument(
        "--solvers", "-s",
        nargs="+",
        choices=["cplex_cp", "cplex_mip", "ortools", "gurobi"],
        default=["cplex_cp", "cplex_mip", "ortools", "gurobi"],
        help="Solvers to benchmark (default: all)",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional list of specific instance names or patterns to run",
    )
    args = parser.parse_args()

    grouped_files = find_instance_files(args.input_dir, args.category)

    # Filter specific instances if requested
    if args.instances:
        for cat in list(grouped_files.keys()):
            filtered = []
            for f in grouped_files[cat]:
                stem = get_instance_stem(f)
                bname = os.path.basename(f)
                if stem in args.instances or bname in args.instances:
                    filtered.append(f)
            grouped_files[cat] = filtered

    total_instances = sum(len(files) for files in grouped_files.values())
    if total_instances == 0:
        print(f"No matching instance files found in '{args.input_dir}'.")
        sys.exit(1)

    print("=" * 90)
    print("BANDWIDTH MINIMIZATION BENCHMARK — BATCH RUNNER")
    print("=" * 90)
    print(f"Input Directory  : {args.input_dir}")
    print(f"Output Directory : {args.output_dir}")
    print(f"Categories       : {', '.join(grouped_files.keys())}")
    for cat, files in grouped_files.items():
        print(f"  - {cat:<18}: {len(files)} instances")
    print(f"Active Solvers   : {', '.join(args.solvers)}")
    print(f"Time Limit       : {args.time_limit}s")
    print("=" * 90 + "\n")

    registry = get_solver_registry()
    csv_headers = [
        "instance",
        "n",
        "edges",
        "lb",
        "solver",
        "bandwidth",
        "encoding_time",
        "solve_time",
    ]

    for category, file_list in grouped_files.items():
        if not file_list:
            continue

        cat_output_dir = os.path.join(args.output_dir, category)
        os.makedirs(cat_output_dir, exist_ok=True)

        print(f">>> Processing category: {category} ({len(file_list)} instances)")
        print(f"    Output folder: {cat_output_dir}")

        # Set up per-solver CSV writers and summary writer
        csv_handles = {}
        csv_writers = {}
        for solver_key in args.solvers:
            display_name, _ = registry[solver_key]
            safe_name = display_name.replace(" ", "_").replace("-", "_")
            csv_path = os.path.join(cat_output_dir, f"{safe_name}.csv")
            fh = open(csv_path, mode="w", newline="", encoding="utf-8", buffering=1)
            writer = csv.DictWriter(fh, fieldnames=csv_headers)
            writer.writeheader()
            csv_handles[solver_key] = fh
            csv_writers[solver_key] = writer

        # Consolidated summary CSV for this category
        summary_path = os.path.join(cat_output_dir, "results.csv")
        summary_fh = open(summary_path, mode="w", newline="", encoding="utf-8", buffering=1)
        summary_writer = csv.DictWriter(summary_fh, fieldnames=csv_headers)
        summary_writer.writeheader()

        try:
            for idx, filepath in enumerate(file_list, start=1):
                instance_stem = get_instance_stem(filepath)
                print(f"\n  [{idx}/{len(file_list)}] Instance '{instance_stem}' ({filepath})")

                try:
                    n, edges = load_hb_file(filepath)
                except Exception as e:
                    print(f"    Error reading file: {e}")
                    for solver_key in args.solvers:
                        display_name, _ = registry[solver_key]
                        err_row = {
                            "instance": instance_stem,
                            "n": "",
                            "edges": "",
                            "lb": "",
                            "solver": display_name,
                            "bandwidth": "",
                            "encoding_time": "",
                            "solve_time": "",
                        }
                        csv_writers[solver_key].writerow(err_row)
                        summary_writer.writerow(err_row)
                    continue

                original_bw = compute_original_bandwidth(n, edges)
                lb_info = compute_caprara_salazar_lb(n, edges)
                lb_parts = [f"deg:{lb_info.get('lb_deg', 0)}"]
                if "alpha" in lb_info:
                    lb_parts.append(f"alpha:{lb_info['alpha']}")
                if "gamma" in lb_info:
                    lb_parts.append(f"gamma:{lb_info['gamma']}")
                print(
                    f"    n={n}, |E|={len(edges)}, Init BW={original_bw}, "
                    f"LB={lb_info['lb']} ({', '.join(lb_parts)})"
                )

                for solver_key in args.solvers:
                    display_name, solver_func = registry[solver_key]
                    if solver_func is None:
                        err_row = {
                            "instance": instance_stem,
                            "n": n,
                            "edges": len(edges),
                            "lb": lb_info["lb"],
                            "solver": display_name,
                            "bandwidth": "",
                            "encoding_time": "",
                            "solve_time": "",
                        }
                        csv_writers[solver_key].writerow(err_row)
                        summary_writer.writerow(err_row)
                        continue

                    print(f"    Running {display_name}...", end="", flush=True)
                    try:
                        res = solver_func(
                            n,
                            edges,
                            time_limit=args.time_limit,
                            lower_bound=lb_info.get("lb"),
                        )
                        verify_solution(n, edges, res)

                        bw = res.get("bandwidth")
                        enc_time = res.get("encoding_time")
                        sol_time = res.get("solve_time")

                        row = {
                            "instance": instance_stem,
                            "n": n,
                            "edges": len(edges),
                            "lb": lb_info["lb"],
                            "solver": display_name,
                            "bandwidth": bw if bw is not None else "",
                            "encoding_time": round(enc_time, 4) if enc_time is not None else "",
                            "solve_time": round(sol_time, 4) if sol_time is not None else "",
                        }

                        csv_writers[solver_key].writerow(row)
                        summary_writer.writerow(row)

                        enc_s = f"{enc_time:.4f}s" if enc_time is not None else "N/A"
                        sol_s = f"{sol_time:.4f}s" if sol_time is not None else "N/A"
                        bw_s = bw if bw is not None else "-"
                        print(f" Done. BW={bw_s}, Enc={enc_s}, Sol={sol_s}")
                    except Exception as e:
                        print(f" Failed: {type(e).__name__}: {e}")
                        err_row = {
                            "instance": instance_stem,
                            "n": n,
                            "edges": len(edges),
                            "lb": lb_info["lb"],
                            "solver": display_name,
                            "bandwidth": "",
                            "encoding_time": "",
                            "solve_time": "",
                        }
                        csv_writers[solver_key].writerow(err_row)
                        summary_writer.writerow(err_row)

        finally:
            for fh in csv_handles.values():
                fh.close()
            summary_fh.close()

        print(f"\n    ✓ Finished {category}! CSV tables written to '{cat_output_dir}'.\n")

    print(f"All benchmarks complete! Results organized in '{args.output_dir}'.")


if __name__ == "__main__":
    run_benchmark()
