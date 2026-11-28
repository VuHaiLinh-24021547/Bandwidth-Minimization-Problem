"""Pillar 4: Publication-grade Cactus Plot, Solver Comparison, and Ablation Charts."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "Order run", "report_assets")

SOLVER_SPECS: List[Tuple[str, str, str, str]] = [
    ("Order IncrementalSAT", "Order_IncrementalSAT.csv", "#D62728", "o"),
    ("Order MultipleSAT", "Order_MultipleSAT.csv", "#FF7F0E", "s"),
    ("CPLEX CP", "CPLEX_CP.csv", "#1F77B4", "^"),
    ("Gurobi MIP", "Gurobi_MIP.csv", "#2CA02C", "D"),
    ("CPLEX MIP", "CPLEX_MIP.csv", "#9467BD", "v"),
    ("OR-Tools CP-SAT", "OR_Tools_CP_SAT.csv", "#7F7F7F", "P"),
]


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.94,
        "figure.dpi": 300,
    })


def _load_csv(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_cactus_optimal(out_path: str) -> None:
    """Cactus Plot: Cumulative instances solved to proven LB optimality vs. total time (small_instances)."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    for name, fname, color, marker in SOLVER_SPECS:
        rows = _load_csv(os.path.join(OUTPUT_ROOT, "small_instances", fname))
        opt_times = []
        for r in rows:
            if r["bandwidth"] != "" and int(r["bandwidth"]) == int(r["lb"]):
                tot = float(r["encoding_time"]) + float(r["solve_time"])
                opt_times.append(max(tot, 0.004))
        opt_times.sort()
        xs = np.arange(1, len(opt_times) + 1)
        lw = 2.6 if "Order" in name else 1.8
        z = 5 if name == "Order IncrementalSAT" else (4 if name == "Order MultipleSAT" else 3)
        ax.plot(
            xs,
            opt_times,
            label=f"{name} ({len(opt_times)}/33)",
            color=color,
            marker=marker,
            markersize=6.2 if "Order" in name else 5.2,
            linewidth=lw,
            zorder=z,
        )

    ax.set_yscale("log")
    ax.set_xlim(0.5, 14.0)
    ax.set_xticks(range(1, 14))
    ax.set_ylim(0.003, 180)
    ax.axhline(120, color="#444444", linestyle=":", linewidth=1.2, label="Time limit (120s)")
    ax.set_xlabel("Number of Small Instances Solved to Proven Lower Bound (BW = LB)")
    ax.set_ylabel("Total Time per Instance (seconds, log scale)")
    ax.set_title("Cactus Plot: Optimal Solutions on 33 Small Instances (Time Limit = 120s)")
    ax.legend(loc="lower right", fontsize=9.2, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_solver_comparison(out_path: str) -> None:
    """2-panel bar chart comparing Average Bandwidth across all 6 solvers on small & large instances."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    categories = [
        ("small_instances", "Small Instances (33 Graphs, n <= 199)"),
        ("large_instances", "Large Instances (40 Graphs, 209 <= n <= 992)"),
    ]

    for ax, (cat, title) in zip(axes, categories):
        names = []
        avg_bws = []
        colors = []
        for name, fname, color, _ in SOLVER_SPECS:
            rows = _load_csv(os.path.join(OUTPUT_ROOT, cat, fname))
            solved = [int(r["bandwidth"]) for r in rows if r["bandwidth"] != ""]
            names.append(name.replace(" ", "\n"))
            avg_bws.append(float(np.mean(solved)) if solved else 0.0)
            colors.append(color)

        x = np.arange(len(names))
        bars = ax.bar(x, avg_bws, color=colors, width=0.60, edgecolor="#222222", linewidth=0.7)
        for bar, val in zip(bars, avg_bws):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + max(avg_bws) * 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.8)
        ax.set_ylim(0, max(avg_bws) * 1.16)
        ax.set_ylabel("Average Bandwidth (lower is better)")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Average Bandwidth Comparison Across All 6 Exact Solvers (73 Instances, 120s Limit)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_head_to_head_scatter(out_path: str) -> None:
    """Head-to-head scatter plot (all 73 instances): Order IncrementalSAT vs Gurobi MIP & OR-Tools CP-SAT."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    targets = [
        ("Gurobi MIP", "Gurobi_MIP.csv", "#2CA02C"),
        ("OR-Tools CP-SAT", "OR_Tools_CP_SAT.csv", "#1F77B4"),
    ]

    for ax, (other_name, other_fn, color) in zip(axes, targets):
        xs, ys = [], []
        wins = ties = losses = 0
        for cat in ["small_instances", "large_instances"]:
            r_inc = _load_csv(os.path.join(OUTPUT_ROOT, cat, "Order_IncrementalSAT.csv"))
            r_oth = _load_csv(os.path.join(OUTPUT_ROOT, cat, other_fn))
            for a, b in zip(r_inc, r_oth):
                if a["bandwidth"] == "" or b["bandwidth"] == "":
                    continue
                v_inc = int(a["bandwidth"])
                v_oth = int(b["bandwidth"])
                xs.append(v_oth)
                ys.append(v_inc)
                if v_inc < v_oth:
                    wins += 1
                elif v_inc == v_oth:
                    ties += 1
                else:
                    losses += 1

        lo, hi = 2.5, 1150
        ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1.2, label="Tie (y = x)")
        ax.scatter(xs, ys, color=color, alpha=0.82, edgecolors="#111111", linewidths=0.5, s=42, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{other_name} Bandwidth (log)")
        ax.set_ylabel("Order IncrementalSAT Bandwidth (log)")
        ax.set_title(f"Order IncrementalSAT vs {other_name}\n(Win: {wins} | Tie: {ties} | Loss: {losses})")
        ax.text(
            0.96,
            0.05,
            "Below diagonal = Order SAT better",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E7", edgecolor="#CC9900", alpha=0.92),
        )
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Head-to-Head Bandwidth Across All 73 Instances (Lower is Better)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ablation_and_amo(out_path: str) -> None:
    """2-panel Ablation chart: (Left) 5-config progression; (Right) Pairwise O(n^3) vs Sinz O(n^2) AMO scaling."""
    matrix_path = os.path.join(ASSETS_DIR, "ablation_matrix.csv")
    amo_path = os.path.join(ASSETS_DIR, "amo_clause_scaling.csv")
    if not os.path.exists(matrix_path) or not os.path.exists(amo_path):
        print("Skipping ablation chart (run ablation_study.py first).")
        return

    rows = _load_csv(matrix_path)
    amo_rows = _load_csv(amo_path)

    cfg_order = ["C1_Naive", "C2_+RCM", "C3_+SymDeg", "C4_+Seq2Hop", "C5_+IncrAssump"]
    cfg_labels = ["C1\nNaive", "C2\n+RCM", "C3\n+Sym/Deg", "C4\n+Seq/2-hop", "C5\n+Incr/Act_k"]
    avg_tot = []
    opt_counts = []
    total_inst = len({r["instance"] for r in rows})
    for c in cfg_order:
        sub = [r for r in rows if r["config"] == c]
        avg_tot.append(float(np.mean([float(r["total_time"]) for r in sub])))
        opt_counts.append(sum(int(r["hit_lb"]) for r in sub))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    x = np.arange(len(cfg_order))
    bar_colors = ["#8C564B", "#7F7F7F", "#1F77B4", "#FF7F0E", "#D62728"]
    bars = ax1.bar(x, avg_tot, color=bar_colors, width=0.58, edgecolor="#222222", linewidth=0.7)
    for bar, val, oc in zip(bars, avg_tot, opt_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(avg_tot) * 0.025,
            f"{val:.2f}s\n({oc}/{total_inst} Opt)",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_labels, fontsize=9.2)
    ax1.set_ylim(0, max(avg_tot) * 1.26)
    ax1.set_ylabel("Average Total Time (s, lower is better)")
    ax1.set_title(f"Ablation Study Progression ({total_inst} Representative Graphs)")
    ax1.grid(axis="x", visible=False)

    ns = [int(r["n"]) for r in amo_rows]
    pw = [int(r["pairwise_amo_clauses"]) / 1e6 for r in amo_rows]
    sc = [int(r["seqcounter_amo_clauses"]) / 1e6 for r in amo_rows]
    ax2.plot(ns, pw, marker="s", color="#7F7F7F", linewidth=2.2, label="Pairwise AMO $O(n^3)$")
    ax2.plot(ns, sc, marker="o", color="#D62728", linewidth=2.5, label="Sequential Counter AMO $O(n^2)$")
    ax2.fill_between(ns, sc, pw, color="#FFCCCC", alpha=0.35, label="Clause Reduction")
    ax2.annotate(
        f"-{amo_rows[-1]['reduction_pct']}% clauses\nat n={ns[-1]}",
        xy=(ns[-1], sc[-1]),
        xytext=(128, 0.72),
        arrowprops=dict(arrowstyle="->", color="#D62728", lw=1.4),
        fontsize=9.2,
        fontweight="bold",
        color="#B22222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D62728"),
    )
    ax2.set_xlabel("Number of Vertices (n)")
    ax2.set_ylabel("AMO Clauses (Millions)")
    ax2.set_title("AMO Clause Scaling: Pairwise $O(n^3)$ vs Sinz $O(n^2)$")
    ax2.legend(loc="upper left", fontsize=9.0)

    fig.suptitle("Order Encoding Ablation & Clause Complexity Reduction", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    _setup_style()

    p1 = os.path.join(ASSETS_DIR, "cactus_plot_optimal.png")
    p2 = os.path.join(ASSETS_DIR, "solver_comparison_bw.png")
    p3 = os.path.join(ASSETS_DIR, "head_to_head_scatter.png")
    p4 = os.path.join(ASSETS_DIR, "ablation_progression.png")

    plot_cactus_optimal(p1)
    plot_solver_comparison(p2)
    plot_head_to_head_scatter(p3)
    plot_ablation_and_amo(p4)

    for p in (p1, p2, p3, p4):
        if os.path.exists(p):
            print(f"Generated chart: {p}")


if __name__ == "__main__":
    main()
