from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hb_parser import load_hb_file
from utils.lb_loader import (
    get_instance_category,
    get_instance_stem,
    load_lb_mapping,
    lookup_lb,
)
from solver.SAT import multipleSAT, incrementalSAT


SOLVER_REGISTRY = {
    "order_incremental_sat": ("Order IncrementalSAT", incrementalSAT.solve),
    "order_multiple_sat": ("Order MultipleSAT", multipleSAT.solve),
}

CSV_HEADERS = [
    "instance",
    "n",
    "edges",
    "lb",
    "solver",
    "bandwidth",
    "encoding_time",
    "solve_time",
]


def find_instances(input_dir: str, category: str = "small_instances") -> List[str]:
    """Find and sort instances by vertex count."""
    if category == "all":
        pattern = os.path.join(input_dir, "**", "*.mtx.rnd")
    else:
        pattern = os.path.join(input_dir, category, "*.mtx.rnd")

    files = glob.glob(pattern, recursive=True)
    if not files:
        fallback_pat = pattern.replace(".mtx.rnd", ".mtx")
        files = glob.glob(fallback_pat, recursive=True)

    def get_vertex_count(f: str) -> int:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    parts = line.split()
                    if len(parts) == 3 and parts[0].isdigit():
                        return int(parts[0])
        except Exception:
            pass
        return 999999

    files.sort(key=get_vertex_count)
    return files


def run_benchmark(
    category: str = "small_instances",
    time_limit: int = 15,
    solvers: Optional[List[str]] = None,
    instances: Optional[List[str]] = None,
    limit: Optional[int] = None,
    input_dir: str = os.path.join(PROJECT_ROOT, "input"),
    output_dir: str = os.path.join(PROJECT_ROOT, "output"),
    lb_csv: str = os.path.join(PROJECT_ROOT, "input", "lb.csv"),
):
    """Run Order Encoding SAT benchmarks and export results directly to output folder."""
    if solvers is None:
        solvers = ["order_incremental_sat", "order_multiple_sat"]

    lb_table = load_lb_mapping(lb_csv)
    file_list = find_instances(input_dir, category)

    if instances:
        file_list = [
            f for f in file_list
            if get_instance_stem(f) in instances or os.path.basename(f) in instances
        ]

    if limit is not None and limit > 0:
        file_list = file_list[:limit]

    if not file_list:
        print(f"No instances found for category '{category}' in '{input_dir}'.")
        return

    cat_output_dir = os.path.join(output_dir, category)
    os.makedirs(cat_output_dir, exist_ok=True)

    print("=" * 96)
    print(f"ORDER ENCODING BENCHMARK — Category: {category} ({len(file_list)} instances)")
    print(f"Time limit: {time_limit}s | Solvers: {', '.join(solvers)}")
    print(f"Output directory: {cat_output_dir}")
    print("=" * 96)
    print(f"{'Instance':<14} | {'n':<4} | {'|E|':<5} | {'LB':<4} | " +
          " | ".join(f"{SOLVER_REGISTRY[s][0]:<22}" for s in solvers))
    print("-" * 96)

    # Initialize CSV writers
    csv_handles = {}
    csv_writers = {}
    for sk in solvers:
        disp_name, _ = SOLVER_REGISTRY[sk]
        safe_name = disp_name.replace(" ", "_").replace("-", "_")
        csv_file = os.path.join(cat_output_dir, f"{safe_name}.csv")
        fh = open(csv_file, mode="w", newline="", encoding="utf-8", buffering=1)
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        writer.writeheader()
        csv_handles[sk] = fh
        csv_writers[sk] = writer

    try:
        for idx, fpath in enumerate(file_list, start=1):
            stem = get_instance_stem(fpath)
            try:
                n, edges = load_hb_file(fpath)
            except Exception as e:
                print(f"Failed to load {fpath}: {e}")
                continue

            lb = lookup_lb(stem, lb_table)
            lb_str = str(lb) if lb is not None else "-"

            row_summary = []
            for sk in solvers:
                disp_name, solver_fn = SOLVER_REGISTRY[sk]
                res = solver_fn(n, edges, time_limit=time_limit, lower_bound=lb)

                bw = res.get("bandwidth", "")
                enc_t = res.get("encoding_time", 0.0)
                slv_t = res.get("solve_time", 0.0)
                tot_t = (enc_t if enc_t else 0.0) + (slv_t if slv_t else 0.0)
                status = res.get("status", "")

                csv_row = {
                    "instance": stem,
                    "n": n,
                    "edges": len(edges),
                    "lb": lb if lb is not None else "",
                    "solver": disp_name,
                    "bandwidth": bw if bw is not None else "",
                    "encoding_time": f"{enc_t:.4f}" if enc_t is not None else "",
                    "solve_time": f"{slv_t:.4f}" if slv_t is not None else "",
                }
                csv_writers[sk].writerow(csv_row)

                bw_disp = str(bw) if bw is not None else "-"
                row_summary.append(f"BW={bw_disp:<3} ({status:<8} {tot_t:5.2f}s)")

            summary_str = " | ".join(row_summary)
            print(f"{stem:<14} | {n:<4} | {len(edges):<5} | {lb_str:<4} | {summary_str}")

    finally:
        for fh in csv_handles.values():
            fh.close()

    print("=" * 96)
    print("Benchmark completed. Results exported directly to:")
    for sk in solvers:
        disp_name, _ = SOLVER_REGISTRY[sk]
        safe_name = disp_name.replace(" ", "_").replace("-", "_")
        print(f"  - {os.path.join(cat_output_dir, f'{safe_name}.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark runner for Order Encoding SAT solvers.")
    parser.add_argument(
        "--category", "-c",
        type=str,
        default="small_instances",
        choices=["small_instances", "large_instances", "all"],
        help="Instance category (default: small_instances)",
    )
    parser.add_argument(
        "--time-limit", "-t",
        type=int,
        default=15,
        help="Per-instance time limit in seconds (default: 15)",
    )
    parser.add_argument(
        "--solvers", "-s",
        nargs="+",
        choices=["order_incremental_sat", "order_multiple_sat"],
        default=["order_incremental_sat", "order_multiple_sat"],
        help="Solvers to benchmark",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional list of specific instance stems",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of instances to benchmark",
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=os.path.join(PROJECT_ROOT, "input"),
        help="Input folder",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output"),
        help="Output folder",
    )
    args = parser.parse_args()

    run_benchmark(
        category=args.category,
        time_limit=args.time_limit,
        solvers=args.solvers,
        instances=args.instances,
        limit=args.limit,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
