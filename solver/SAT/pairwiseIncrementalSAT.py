"""Incremental SAT solver isolated for Pairwise Encoding."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_pairwise_encoder():
    import encoder.pairwise as pairwise_mod
    encoder_cls = getattr(pairwise_mod, "PairwiseEncoder", None)
    if encoder_cls is None:
        raise ImportError(
            "PairwiseEncoder is not yet implemented in 'encoder/pairwise.py'."
        )
    return encoder_cls


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
) -> Dict:
    PairwiseEncoder = _load_pairwise_encoder()
    from pysat.solvers import Solver

    total_encoding_time = 0.0
    total_solve_time = 0.0
    start_wall_time = time.perf_counter()
    deadline = start_wall_time + float(time_limit)

    lb = max(1, lower_bound) if lower_bound is not None and lower_bound > 0 else 1

    best_perm = list(range(n))
    is_zero = any(u == 0 or v == 0 for u, v in edges) if edges else True
    if not is_zero and edges and max(max(u, v) for u, v in edges) < n:
        is_zero = True
    best_bw = max(
        abs(best_perm[u] - best_perm[v]) if is_zero else abs(best_perm[u - 1] - best_perm[v - 1])
        for u, v in edges
    ) if edges else 0
    status = "Feasible"

    if best_bw <= lb:
        return {
            "solver": "Pairwise IncrementalSAT",
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    t_enc0 = time.perf_counter()
    encoder = PairwiseEncoder(n, edges)
    base_clauses = encoder.get_base_clauses()
    total_encoding_time += time.perf_counter() - t_enc0

    if time.perf_counter() >= deadline:
        return {
            "solver": "Pairwise IncrementalSAT",
            "status": "TimeLimitReached",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    t_boot0 = time.perf_counter()
    solver = Solver(name=solver_name, bootstrap_with=base_clauses)
    total_encoding_time += time.perf_counter() - t_boot0

    try:
        k = best_bw - 1
        while k >= lb:
            if time.perf_counter() >= deadline:
                status = "TimeLimitReached"
                break

            t_delta0 = time.perf_counter()
            bw_clauses = encoder.get_bandwidth_clauses(k)
            solver.append_formula(bw_clauses)
            total_encoding_time += time.perf_counter() - t_delta0

            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                status = "TimeLimitReached"
                break

            timer = threading.Timer(max(0.01, remaining_time), solver.interrupt)
            timer.start()

            t_s0 = time.perf_counter()
            sat = None
            try:
                sat = solver.solve_limited(expect_interrupt=True)
            finally:
                timer.cancel()
                solver.clear_interrupt()
                total_solve_time += time.perf_counter() - t_s0

            if sat is None or time.perf_counter() >= deadline:
                status = "TimeLimitReached"
                break

            if sat is True:
                perm = encoder.decode_permutation(solver.get_model())
                actual_bw = max(
                    abs(perm[u] - perm[v]) if is_zero else abs(perm[u - 1] - perm[v - 1])
                    for u, v in edges
                ) if edges else 0

                if actual_bw <= best_bw:
                    best_bw = actual_bw
                    best_perm = perm

                if best_bw <= lb:
                    status = "Optimal"
                    break

                k = best_bw - 1
            else:
                status = "Optimal"
                break

        if k < lb and status != "TimeLimitReached":
            status = "Optimal"
    finally:
        solver.delete()

    return {
        "solver": "Pairwise IncrementalSAT",
        "status": status,
        "bandwidth": best_bw,
        "permutation": best_perm,
        "encoding_time": total_encoding_time,
        "solve_time": total_solve_time,
    }
