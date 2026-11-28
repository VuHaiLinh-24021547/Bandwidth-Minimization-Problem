"""Pillar 4: Ablation Study for Order Encoding (5 progressive configurations + AMO scaling)."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.SAT import orderIncrementalSAT, orderMultipleSAT
from utils.hb_parser import load_hb_file
from utils.lb_loader import load_lb_mapping, lookup_lb, verify_solution

DEFAULT_INSTANCES = [
    "pores_1",
    "ibm32",
    "bcspwr01",
    "bcsstk01",
    "bcspwr02",
    "curtis54",
    "will57",
    "ash85",
    "nos4",
    "can__144",
]

ABLATION_CONFIGS: List[Tuple[str, str, Dict]] = [
    (
        "C1_Naive",
        "multiple",
        dict(
            use_rcm=False,
            use_symmetry_breaking=False,
            use_degree_bounds=False,
            use_strengthening=False,
            use_2hop=False,
            amo_encoding="pairwise",
        ),
    ),
    (
        "C2_+RCM",
        "multiple",
        dict(
            use_rcm=True,
            use_symmetry_breaking=False,
            use_degree_bounds=False,
            use_strengthening=False,
            use_2hop=False,
            amo_encoding="pairwise",
        ),
    ),
    (
        "C3_+SymDeg",
        "multiple",
        dict(
            use_rcm=True,
            use_symmetry_breaking=True,
            use_degree_bounds=True,
            use_strengthening=True,
            use_2hop=False,
            amo_encoding="pairwise",
        ),
    ),
    (
        "C4_+Seq2Hop",
        "multiple",
        dict(
            use_rcm=True,
            use_symmetry_breaking=True,
            use_degree_bounds=True,
            use_strengthening=True,
            use_2hop=True,
            amo_encoding="seqcounter",
        ),
    ),
    (
        "C5_+IncrAssump",
        "incremental",
        dict(
            use_rcm=True,
            use_symmetry_breaking=True,
            use_degree_bounds=True,
            use_strengthening=True,
            use_2hop=True,
            amo_encoding="seqcounter",
            search_strategy="binary",
        ),
    ),
]


def export_amo_scaling(out_csv: str) -> None:
    """Export Pairwise O(n^3) vs Sinz Sequential Counter O(n^2) AMO clause counts."""
    ns = [30, 50, 85, 100, 144, 180, 200]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["n", "pairwise_amo_clauses", "seqcounter_amo_clauses", "reduction_pct"],
        )
        writer.writeheader()
        for n in ns:
            pw = n * (n * (n - 1) // 2)
            sc = n * (3 * n - 4)
            red = (1.0 - sc / pw) * 100.0
            writer.writerow({
                "n": n,
                "pairwise_amo_clauses": pw,
                "seqcounter_amo_clauses": sc,
                "reduction_pct": f"{red:.2f}",
            })


def run_ablation(
    instances: List[str],
    time_limit: int = 20,
    output_dir: str = os.path.join(PROJECT_ROOT, "Order run", "report_assets"),
) -> None:
    """Run all 5 ablation configurations and export matrix CSV + Markdown summary."""
    os.makedirs(output_dir, exist_ok=True)
    lb_table = load_lb_mapping(os.path.join(PROJECT_ROOT, "input", "lb.csv"))

    amo_csv = os.path.join(output_dir, "amo_clause_scaling.csv")
    export_amo_scaling(amo_csv)

    matrix_csv = os.path.join(output_dir, "ablation_matrix.csv")
    summary_md = os.path.join(output_dir, "ablation_summary.md")

    rows: List[Dict] = []
    print("=" * 106, flush=True)
    print(f"ABLATION STUDY ({len(instances)} instances x {len(ABLATION_CONFIGS)} configs, time_limit={time_limit}s)", flush=True)
    print("=" * 106, flush=True)

    for stem in instances:
        fpath = os.path.join(PROJECT_ROOT, "input", "small_instances", f"{stem}.mtx.rnd")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Instance file not found: {fpath}")
        n, edges = load_hb_file(fpath)
        lb = lookup_lb(stem, lb_table)

        pw_base = n * (n - 2) + n * (3 * (n - 2) + 4) + 2 * n + n * (n * (n - 1) // 2)
        sc_base = n * (n - 2) + n * (3 * (n - 2) + 4) + 2 * n + n * (3 * n - 4)

        for cfg_name, mode, kwargs in ABLATION_CONFIGS:
            solver_fn = orderIncrementalSAT.solve if mode == "incremental" else orderMultipleSAT.solve
            res = solver_fn(n, edges, time_limit=time_limit, lower_bound=lb, **kwargs)
            verify_solution(n, edges, res)

            bw = res.get("bandwidth")
            enc_t = float(res.get("encoding_time") or 0.0)
            sol_t = float(res.get("solve_time") or 0.0)
            tot_t = enc_t + sol_t
            status = res.get("status", "")
            hit_lb = (bw is not None and lb is not None and int(bw) == int(lb))
            base_clauses = sc_base if kwargs.get("amo_encoding") == "seqcounter" else pw_base

            row = {
                "instance": stem,
                "n": n,
                "edges": len(edges),
                "lb": lb if lb is not None else "",
                "config": cfg_name,
                "base_clauses": base_clauses,
                "bandwidth": bw if bw is not None else "",
                "hit_lb": int(hit_lb),
                "status": status,
                "encoding_time": f"{enc_t:.4f}",
                "solve_time": f"{sol_t:.4f}",
                "total_time": f"{tot_t:.4f}",
            }
            rows.append(row)
            print(
                f"{stem:<10} (n={n:<3}) | {cfg_name:<15} | base={base_clauses:>9,} | "
                f"BW={str(bw):<3} (LB={lb}) | enc={enc_t:5.2f}s sol={sol_t:5.2f}s | {status}",
                flush=True,
            )
            gc.collect()

    with open(matrix_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "instance", "n", "edges", "lb", "config", "base_clauses",
                "bandwidth", "hit_lb", "status", "encoding_time", "solve_time", "total_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ablation Study Summary — Order Encoding (5 Configurations)",
        "",
        f"Instances ({len(instances)}): `{', '.join(instances)}` | Time limit: `{time_limit}s`",
        "",
        "| Configuration | Optimal (`BW==LB`) | Avg BW | Avg Base Clauses | Avg Enc (s) | Avg Sol (s) | Avg Total (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cfg_name, _, _ in ABLATION_CONFIGS:
        sub = [r for r in rows if r["config"] == cfg_name]
        opt_cnt = sum(r["hit_lb"] for r in sub)
        avg_bw = sum(int(r["bandwidth"]) for r in sub) / len(sub)
        avg_cls = sum(int(r["base_clauses"]) for r in sub) / len(sub)
        avg_enc = sum(float(r["encoding_time"]) for r in sub) / len(sub)
        avg_sol = sum(float(r["solve_time"]) for r in sub) / len(sub)
        avg_tot = sum(float(r["total_time"]) for r in sub) / len(sub)
        lines.append(
            f"| **{cfg_name}** | {opt_cnt}/{len(sub)} | {avg_bw:.2f} | {avg_cls:,.0f} | "
            f"{avg_enc:.2f} | {avg_sol:.2f} | {avg_tot:.2f} |"
        )

    with open(summary_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("=" * 106, flush=True)
    print(f"Saved: {matrix_csv}")
    print(f"Saved: {amo_csv}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-config Ablation Study for Order Encoding.")
    parser.add_argument("--instances", nargs="*", default=DEFAULT_INSTANCES)
    parser.add_argument("--time-limit", "-t", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Order run", "report_assets"),
    )
    args = parser.parse_args()
    run_ablation(args.instances, time_limit=args.time_limit, output_dir=args.output_dir)
