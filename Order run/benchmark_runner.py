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

from solver.SAT import orderIncrementalSAT, orderMultipleSAT
from utils.hb_parser import load_hb_file
from utils.lb_loader import (
    get_instance_category,
    get_instance_stem,
    load_lb_mapping,
    lookup_lb,
    verify_solution,
)


SOLVER_REGISTRY = {
    "order_incremental_sat": ("Order IncrementalSAT", orderIncrementalSAT.solve),
    "order_multiple_sat": ("Order MultipleSAT", orderMultipleSAT.solve),
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


def find_instances(input_dir: str, category: str = "small_instances") -> Dict[str, List[str]]:
    if category == "all":
        categories = ["small_instances", "large_instances"]
    else:
        categories = [category]

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

    grouped: Dict[str, List[str]] = {}
    for cat in categories:
        pattern = os.path.join(input_dir, cat, "*.mtx.rnd")
        files = glob.glob(pattern)
        if not files:
            files = glob.glob(pattern.replace(".mtx.rnd", ".mtx"))
        files.sort(key=get_vertex_count)
        if files:
            grouped[cat] = files
    return grouped


def run_benchmark(
    category: str = "small_instances",
    time_limit: int = 60,
    solvers: Optional[List[str]] = None,
    instances: Optional[List[str]] = None,
    limit: Optional[int] = None,
    input_dir: str = os.path.join(PROJECT_ROOT, "input"),
    output_dir: str = os.path.join(PROJECT_ROOT, "output"),
    lb_csv: str = os.path.join(PROJECT_ROOT, "input", "lb.csv"),
):
    if solvers is None:
        solvers = ["order_incremental_sat", "order_multiple_sat"]

    lb_table = load_lb_mapping(lb_csv)
    grouped = find_instances(input_dir, category)

    if instances:
        for cat in list(grouped.keys()):
            grouped[cat] = [
                f for f in grouped[cat]
                if get_instance_stem(f) in instances or os.path.basename(f) in instances
            ]

    if limit is not None and limit > 0:
        for cat in list(grouped.keys()):
            grouped[cat] = grouped[cat][:limit]

    if not any(grouped.values()):
        print(f"No instances found for category '{category}' in '{input_dir}'.")
        return

    for cat, file_list in grouped.items():
        if not file_list:
            continue

        cat_output_dir = os.path.join(output_dir, cat)
        os.makedirs(cat_output_dir, exist_ok=True)

        print("=" * 96)
        print(f"ORDER ENCODING BENCHMARK — Category: {cat} ({len(file_list)} instances)")
        print(f"Time limit: {time_limit}s | Solvers: {', '.join(solvers)}")
        print(f"Output directory: {cat_output_dir}")
        print("=" * 96)
        print(
            f"{'Instance':<14} | {'n':<4} | {'|E|':<5} | {'LB':<4} | "
            + " | ".join(f"{SOLVER_REGISTRY[s][0]:<22}" for s in solvers)
        )
        print("-" * 96)

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
            for fpath in file_list:
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
                    verify_solution(n, edges, res)

                    bw = res.get("bandwidth", "")
                    enc_t = res.get("encoding_time", 0.0)
                    slv_t = res.get("solve_time", 0.0)
                    tot_t = (enc_t if enc_t else 0.0) + (slv_t if slv_t else 0.0)
                    status = res.get("status", "")

                    csv_writers[sk].writerow({
                        "instance": stem,
                        "n": n,
                        "edges": len(edges),
                        "lb": lb if lb is not None else "",
                        "solver": disp_name,
                        "bandwidth": bw if bw is not None else "",
                        "encoding_time": f"{enc_t:.4f}" if enc_t is not None else "",
                        "solve_time": f"{slv_t:.4f}" if slv_t is not None else "",
                    })

                    bw_disp = str(bw) if bw is not None else "-"
                    row_summary.append(f"BW={bw_disp:<3} ({status:<8} {tot_t:5.2f}s)")

                print(f"{stem:<14} | {n:<4} | {len(edges):<5} | {lb_str:<4} | " + " | ".join(row_summary))
        finally:
            for fh in csv_handles.values():
                fh.close()

        print("=" * 96)
        print("Exported:")
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
    )
    parser.add_argument("--time-limit", "-t", type=int, default=60)
    parser.add_argument(
        "--solvers", "-s",
        nargs="+",
        choices=["order_incremental_sat", "order_multiple_sat"],
        default=["order_incremental_sat", "order_multiple_sat"],
    )
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--input-dir", "-i", type=str, default=os.path.join(PROJECT_ROOT, "input"))
    parser.add_argument("--output-dir", "-o", type=str, default=os.path.join(PROJECT_ROOT, "output"))
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
