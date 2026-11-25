from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hb_parser import load_hb_file
from utils.lb_loader import (
    compute_original_bandwidth,
    get_instance_category,
    get_instance_stem,
    load_lb_mapping,
    lookup_lb,
    verify_solution,
)


def format_results_table(
    n: int,
    num_edges: int,
    original_bw: int,
    results: List[Dict],
    lb: Optional[int] = None,
) -> str:
    """Return a human-readable comparison table as a string."""
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("BANDWIDTH MINIMIZATION PROBLEM — BENCHMARK RESULTS")
    lines.append("=" * 100)
    lines.append(f"  Vertices (n)         : {n}")
    lines.append(f"  Edges    (|E|)       : {num_edges}")
    lines.append(f"  Original bandwidth   : {original_bw}")
    lines.append(f"  Lower Bound (LB)     : {lb if lb is not None else 'N/A'}")

    lb_val = str(lb) if lb is not None else "-"
    lines.append("-" * 85)

    header = (
        f"{'Solver':<18} {'LB':>6} {'Bandwidth':>10} "
        f"{'Enc. Time (s)':>14} {'Solve Time (s)':>15}"
    )
    lines.append(header)
    lines.append("-" * 85)

    for r in results:
        solver = r.get("solver", "?")
        bw = r.get("bandwidth")
        bw_str = str(bw) if bw is not None else "-"
        enc_time = r.get("encoding_time")
        enc_str = f"{enc_time:.3f}" if enc_time is not None else "-"
        solve_time = r.get("solve_time")
        solve_str = f"{solve_time:.3f}" if solve_time is not None else "-"

        lines.append(
            f"{solver:<18} {lb_val:>6} {bw_str:>10} "
            f"{enc_str:>14} {solve_str:>15}"
        )

    lines.append("-" * 85)
    lines.append("=" * 100)
    return "\n".join(lines)


def _run_solver(
    solver_name: str,
    solver_func,
    n: int,
    edges,
    time_limit: int,
    lower_bound: Optional[int] = None,
) -> dict:
    """Run a single solver with error handling."""
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
             "If omitted, a default instance is selected.",
    )
    parser.add_argument(
        "--time-limit", "-t",
        type=int,
        default=120,
        help="Time limit per solver in seconds (default: 120).",
    )
    parser.add_argument(
        "--solvers", "-s",
        nargs="+",
        choices=["cplex_cp", "cplex_mip", "ortools", "gurobi", "pairwise_multiple_sat", "pairwise_incremental_sat"],
        default=["cplex_cp", "cplex_mip", "ortools", "gurobi"],
        help="Which solvers to run (default: all four).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output"),
        help="Output folder path where category folders are stored (default: output).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable appending results to CSV files in output folder.",
    )
    parser.add_argument(
        "--lb-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "input", "lb.csv"),
        help="Path to CSV containing precomputed lower bounds (default: input/lb.csv).",
    )
    args = parser.parse_args()

    # ---- Load or check input ----
    if args.file is not None:
        filepath = args.file
    else:
        filepath = os.path.join(PROJECT_ROOT, "input", "small_instances", "bcspwr01.mtx.rnd")
        if not os.path.exists(filepath):
            candidates = glob.glob(os.path.join(PROJECT_ROOT, "input", "**", "*.mtx.rnd"), recursive=True)
            if candidates:
                filepath = candidates[0]
            else:
                print("Error: No input file specified and none found in input/.")
                sys.exit(1)
        print(f"No input file specified — using default: {filepath}")

    print(f"\nLoading: {filepath}")
    n, edges = load_hb_file(filepath)
    original_bw = compute_original_bandwidth(n, edges)
    instance_stem = get_instance_stem(filepath)

    # ---- Load Lower Bound from CSV ----
    lb_table = load_lb_mapping(args.lb_csv)
    lb = lookup_lb(instance_stem, lb_table)

    print(f"  Instance : {instance_stem}")
    print(f"  Vertices : {n}")
    print(f"  Edges    : {len(edges)}")
    print(f"  Original bandwidth (identity permutation): {original_bw}")
    print(f"  Lower Bound (LB)                         : {lb if lb is not None else 'N/A'}")
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
    if "pairwise_multiple_sat" in args.solvers:
        from solver.SAT import multipleSAT
        registry["Pairwise MultipleSAT"] = multipleSAT.solve
    if "pairwise_incremental_sat" in args.solvers:
        from solver.SAT import incrementalSAT
        registry["Pairwise IncrementalSAT"] = incrementalSAT.solve

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
            lower_bound=lb,
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

    print(format_results_table(n, len(edges), original_bw, results, lb=lb))

    # ---- Export to Category CSV ----
    if not args.no_export:
        category = get_instance_category(filepath)
        cat_output_dir = os.path.join(args.output_dir, category)
        os.makedirs(cat_output_dir, exist_ok=True)

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

        for r in results:
            solver_name = r.get("solver", "?")
            safe_name = solver_name.replace(" ", "_").replace("-", "_")
            csv_path = os.path.join(cat_output_dir, f"{safe_name}.csv")

            write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()

                enc_time = r.get("encoding_time")
                sol_time = r.get("solve_time")
                row = {
                    "instance": instance_stem,
                    "n": n,
                    "edges": len(edges),
                    "lb": lb if lb is not None else "",
                    "solver": solver_name,
                    "bandwidth": r.get("bandwidth") if r.get("bandwidth") is not None else "",
                    "encoding_time": round(enc_time, 4) if enc_time is not None else "",
                    "solve_time": round(sol_time, 4) if sol_time is not None else "",
                }
                writer.writerow(row)

        print(f"\nAppended benchmark results to '{cat_output_dir}/'")


if __name__ == "__main__":
    main()
