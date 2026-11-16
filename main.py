#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the Matrix Bandwidth Minimization Problem (BMP) benchmark.

Usage
-----
    # Run on a specific input file with a 60-second time limit and default results.csv:
    python main.py --file input/small_instances/bcspwr01.mtx.rnd --time-limit 60

    # Custom CSV output path:
    python main.py --file input/small_instances/bcspwr01.mtx.rnd --csv my_results.csv

    # Run only specific solvers:
    python main.py --file input/small_instances/bcspwr01.mtx.rnd --solvers ortools cplex_cp
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

from hb_parser import generate_sample_hb, load_hb_file
from utils import (
    compute_caprara_salazar_lb,
    compute_original_bandwidth,
    compute_theoretical_lb,
    format_results_table,
    verify_solution,
)


def _get_instance_stem(filepath: str) -> str:
    """Extract instance name without .mtx.rnd (or other extensions)."""
    filename = os.path.basename(filepath)
    if filename.endswith(".mtx.rnd"):
        return filename[:-8]
    elif filename.endswith(".rnd"):
        return filename[:-4]
    elif filename.endswith(".mtx"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def _run_solver(solver_name: str, solver_func, n: int, edges, time_limit: int, lower_bound: Optional[int] = None) -> dict:
    """Run a single solver with graceful error handling."""
    try:
        result = solver_func(n, edges, time_limit=time_limit, lower_bound=lower_bound)
    except ImportError as exc:
        print(f"  [{solver_name}] Skipped — missing dependency: {exc}")
        result = {
            "solver": solver_name,
            "status": f"ImportError: {exc}",
            "bandwidth": None,
            "permutation": None,
            "encoding_time": None,
            "solve_time": None,
            "gap": None,
        }
    except Exception as exc:
        print(f"  [{solver_name}] Failed — {type(exc).__name__}: {exc}")
        result = {
            "solver": solver_name,
            "status": f"Error: {type(exc).__name__}",
            "bandwidth": None,
            "permutation": None,
            "encoding_time": None,
            "solve_time": None,
            "gap": None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bandwidth Minimization Problem — Exact Solver Benchmark",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a sparse-matrix file (.mtx.rnd / .mtx / .hb / .rua / .rsa). "
             "If omitted, a synthetic test file is generated.",
    )
    parser.add_argument(
        "--time-limit", "-t",
        type=int,
        default=120,
        help="Time limit per solver in seconds (default: 60).",
    )
    parser.add_argument(
        "--solvers", "-s",
        nargs="+",
        choices=["cplex_cp", "cplex_mip", "ortools", "gurobi"],
        default=["cplex_cp", "cplex_mip", "ortools", "gurobi"],
        help="Which solvers to run (default: all four).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results.csv",
        help="Output CSV file path to append benchmark results (default: results.csv).",
    )
    args = parser.parse_args()

    # ---- Load or generate input ----
    if args.file is not None:
        filepath = args.file
    else:
        filepath = os.path.join("input", "sample_generated.mtx.rnd")
        print(f"No input file specified — generating synthetic test: {filepath}")
        generate_sample_hb(filepath, n=15)

    print(f"\nLoading: {filepath}")
    n, edges = load_hb_file(filepath)
    original_bw = compute_original_bandwidth(n, edges)
    instance_stem = _get_instance_stem(filepath)

    # ---- Compute Graph-Theoretic Lower Bound ----
    lb_info = compute_caprara_salazar_lb(n, edges)

    print(f"  Instance : {instance_stem}")
    print(f"  Vertices : {n}")
    print(f"  Edges    : {len(edges)}")
    print(f"  Original bandwidth (identity permutation): {original_bw}")
    lb_parts = [f"deg: {lb_info['lb_deg']}"]
    if "alpha" in lb_info:
        lb_parts.append(f"alpha: {lb_info['alpha']}")
    if "gamma" in lb_info:
        lb_parts.append(f"gamma: {lb_info['gamma']}")
    print(f"  Lower Bound (LB)                         : {lb_info['lb']} ({', '.join(lb_parts)})")
    print()

    # ---- Solver registry ----
    registry = {}
    if "cplex_cp" in args.solvers:
        from solver.CP import CplexCP
        registry["CPLEX CP"] = CplexCP.solve
    if "cplex_mip" in args.solvers:
        from solver.ILP import CplexMIP
        registry["CPLEX MIP"] = CplexMIP.solve
    if "ortools" in args.solvers:
        from solver.CP import CPSat
        registry["OR-Tools CP-SAT"] = CPSat.solve
    if "gurobi" in args.solvers:
        from solver.ILP import gurobi
        registry["Gurobi MIP"] = gurobi.solve

    # ---- Run solvers ----
    results = []
    for solver_name, solver_func in registry.items():
        print(f"Running {solver_name}...", end="", flush=True)
        result = _run_solver(
            solver_name,
            solver_func,
            n,
            edges,
            args.time_limit,
            lower_bound=lb_info.get("lb"),
        )
        verify_solution(n, edges, result)
        results.append(result)

        bw = result.get("bandwidth")
        t_enc = result.get("encoding_time")
        t_sol = result.get("solve_time")
        enc_str = f"{t_enc:.4f}s" if t_enc is not None else "N/A"
        sol_str = f"{t_sol:.4f}s" if t_sol is not None else "N/A"
        bw_str = str(bw) if bw is not None else "N/A"
        print(f" Done. BW={bw_str}, Enc={enc_str}, Sol={sol_str}")
        print()

    # ---- Print summary table ----
    print(format_results_table(n, len(edges), original_bw, results, lb_info=lb_info))

    # ---- Export to CSV ----
    if args.csv:
        csv_path = args.csv
        parent_dir = os.path.dirname(csv_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        fieldnames = [
            "instance",
            "n",
            "edges",
            "lb",
            "solver",
            "bandwidth",
            "encoding_time",
            "solve_time",
        ]

        write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for r in results:
                enc_time = r.get("encoding_time")
                sol_time = r.get("solve_time")

                row = {
                    "instance": instance_stem,
                    "n": n,
                    "edges": len(edges),
                    "lb": lb_info["lb"],
                    "solver": r.get("solver", "?"),
                    "bandwidth": r.get("bandwidth") if r.get("bandwidth") is not None else "",
                    "encoding_time": round(enc_time, 4) if enc_time is not None else "",
                    "solve_time": round(sol_time, 4) if sol_time is not None else "",
                }
                writer.writerow(row)

        print(f"\nAppended benchmark results to '{csv_path}'")


if __name__ == "__main__":
    main()
