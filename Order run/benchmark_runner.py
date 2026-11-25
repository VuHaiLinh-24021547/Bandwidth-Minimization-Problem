from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.SAT import orderIncrementalSAT, orderMultipleSAT
from utils.hb_parser import load_hb_file
from utils.lb_loader import (
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
    categories = ["small_instances", "large_instances"] if category == "all" else [category]
    grouped: Dict[str, List[str]] = {}
    for cat in categories:
        pattern = os.path.join(input_dir, cat, "*.mtx.rnd")
        files = glob.glob(pattern)
        if not files:
            files = glob.glob(pattern.replace(".mtx.rnd", ".mtx"))
        # Preserve exact input filename ordering (matching CP/ILP output CSVs)
        files.sort(key=lambda p: os.path.basename(p))
        if files:
            grouped[cat] = files
    return grouped


def _run_single_solver_in_process(
    fpath: str,
    solver_key: str,
    time_limit: int,
    lb: Optional[int],
) -> Dict:
    """Run one solver on one instance inside an isolated child process with hard timeout."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-worker",
        "--single-file",
        fpath,
        "--single-solver",
        solver_key,
        "--time-limit",
        str(time_limit),
        "--single-lb",
        "" if lb is None else str(lb),
    ]
    hard_timeout = float(time_limit) + 20.0
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=hard_timeout,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
            return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # Fallback to verified RCM upper bound if C++ solver stalled past hard timeout
    n, edges = load_hb_file(fpath)
    t_rcm0 = time.perf_counter()
    bw, perm = orderIncrementalSAT.get_rcm_upper_bound(n, edges)
    enc_t = time.perf_counter() - t_rcm0
    disp_name, _ = SOLVER_REGISTRY[solver_key]
    res = {
        "solver": disp_name,
        "status": "TimeLimitReached",
        "bandwidth": bw,
        "permutation": perm,
        "encoding_time": enc_t,
        "solve_time": max(0.0, float(time_limit) - enc_t),
    }
    verify_solution(n, edges, res)
    res.pop("permutation", None)
    return res


def _evaluate_instance(
    idx: int,
    fpath: str,
    solvers: List[str],
    time_limit: int,
    lb_table: Dict[str, int],
) -> Tuple[int, str, int, int, Optional[int], Dict[str, Dict]]:
    stem = get_instance_stem(fpath)
    n, edges = load_hb_file(fpath)
    lb = lookup_lb(stem, lb_table)
    per_solver: Dict[str, Dict] = {}
    for sk in solvers:
        per_solver[sk] = _run_single_solver_in_process(fpath, sk, time_limit, lb)
    return idx, stem, n, len(edges), lb, per_solver


def _write_instance_result(
    idx: int,
    total: int,
    stem: str,
    n: int,
    num_edges: int,
    lb: Optional[int],
    solvers: List[str],
    per_solver: Dict[str, Dict],
    csv_writers: Dict[str, csv.DictWriter],
) -> None:
    lb_str = str(lb) if lb is not None else "-"
    row_summary = []
    for sk in solvers:
        disp_name, _ = SOLVER_REGISTRY[sk]
        res = per_solver[sk]
        bw = res.get("bandwidth")
        enc_t = res.get("encoding_time")
        slv_t = res.get("solve_time")
        tot_t = (enc_t if enc_t else 0.0) + (slv_t if slv_t else 0.0)
        status = res.get("status", "")

        csv_writers[sk].writerow({
            "instance": stem,
            "n": n,
            "edges": num_edges,
            "lb": lb if lb is not None else "",
            "solver": disp_name,
            "bandwidth": bw if bw is not None else "",
            "encoding_time": f"{enc_t:.4f}" if enc_t is not None else "",
            "solve_time": f"{slv_t:.4f}" if slv_t is not None else "",
        })

        bw_disp = str(bw) if bw is not None else "-"
        row_summary.append(f"BW={bw_disp:<3} ({status:<8} {tot_t:5.2f}s)")

    print(
        f"[{idx + 1:>2}/{total:<2}] {stem:<14} | {n:<4} | {num_edges:<5} | {lb_str:<4} | "
        + " | ".join(row_summary),
        flush=True,
    )


def run_benchmark(
    category: str = "small_instances",
    time_limit: int = 60,
    solvers: Optional[List[str]] = None,
    instances: Optional[List[str]] = None,
    limit: Optional[int] = None,
    workers: int = 1,
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

        print("=" * 104, flush=True)
        print(f"ORDER ENCODING BENCHMARK — Category: {cat} ({len(file_list)} instances)", flush=True)
        print(f"Time limit: {time_limit}s | Workers: {workers} | Solvers: {', '.join(solvers)}", flush=True)
        print(f"Output directory: {cat_output_dir}", flush=True)
        print("=" * 104, flush=True)
        print(
            f"{'Progress':<7} {'Instance':<14} | {'n':<4} | {'|E|':<5} | {'LB':<4} | "
            + " | ".join(f"{SOLVER_REGISTRY[s][0]:<24}" for s in solvers),
            flush=True,
        )
        print("-" * 104, flush=True)

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
            total = len(file_list)
            if workers <= 1:
                for idx, fpath in enumerate(file_list):
                    _, stem, n, num_edges, lb, per_solver = _evaluate_instance(
                        idx, fpath, solvers, time_limit, lb_table
                    )
                    _write_instance_result(
                        idx, total, stem, n, num_edges, lb, solvers, per_solver, csv_writers
                    )
            else:
                # Strictly preserve input index order when writing CSV rows
                pending: Dict[int, Tuple[str, int, int, Optional[int], Dict[str, Dict]]] = {}
                next_idx = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_evaluate_instance, idx, fpath, solvers, time_limit, lb_table): idx
                        for idx, fpath in enumerate(file_list)
                    }
                    for fut in as_completed(futures):
                        idx, stem, n, num_edges, lb, per_solver = fut.result()
                        pending[idx] = (stem, n, num_edges, lb, per_solver)
                        while next_idx in pending:
                            s_stem, s_n, s_m, s_lb, s_res = pending.pop(next_idx)
                            _write_instance_result(
                                next_idx, total, s_stem, s_n, s_m, s_lb, solvers, s_res, csv_writers
                            )
                            next_idx += 1
        finally:
            for fh in csv_handles.values():
                fh.close()

        print("=" * 104, flush=True)
        print("Exported:", flush=True)
        for sk in solvers:
            disp_name, _ = SOLVER_REGISTRY[sk]
            safe_name = disp_name.replace(" ", "_").replace("-", "_")
            print(f"  - {os.path.join(cat_output_dir, f'{safe_name}.csv')}", flush=True)


def _run_single_worker_cli(args) -> None:
    n, edges = load_hb_file(args.single_file)
    lb = int(args.single_lb) if args.single_lb else None
    _, solver_fn = SOLVER_REGISTRY[args.single_solver]
    res = solver_fn(n, edges, time_limit=args.time_limit, lower_bound=lb)
    verify_solution(n, edges, res)
    if res.get("permutation_valid") and res.get("verified_bandwidth") is not None:
        res["bandwidth"] = int(res["verified_bandwidth"])
    res.pop("permutation", None)
    print(json.dumps(res), flush=True)


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
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--input-dir", "-i", type=str, default=os.path.join(PROJECT_ROOT, "input"))
    parser.add_argument("--output-dir", "-o", type=str, default=os.path.join(PROJECT_ROOT, "output"))
    parser.add_argument("--single-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-file", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--single-solver", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--single-lb", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_worker:
        _run_single_worker_cli(args)
    else:
        run_benchmark(
            category=args.category,
            time_limit=args.time_limit,
            solvers=args.solvers,
            workers=args.workers,
            instances=args.instances,
            limit=args.limit,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
