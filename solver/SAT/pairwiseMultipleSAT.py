"""Multiple SAT solver isolated for Pairwise Encoding."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.SAT.pairwiseIncrementalSAT import _load_pairwise_encoder


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
            "solver": "Pairwise MultipleSAT",
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    t_init0 = time.perf_counter()
    encoder = PairwiseEncoder(n, edges)
    total_encoding_time += time.perf_counter() - t_init0

    low = lb
    high = best_bw - 1

    while low <= high:
        if time.perf_counter() >= deadline:
            status = "TimeLimitReached"
            break

        mid = (low + high) // 2

        t_enc0 = time.perf_counter()
        clauses = encoder.encode(mid)
        total_encoding_time += time.perf_counter() - t_enc0

        if time.perf_counter() >= deadline:
            status = "TimeLimitReached"
            break

        t_boot0 = time.perf_counter()
        solver = Solver(name=solver_name, bootstrap_with=clauses)
        total_encoding_time += time.perf_counter() - t_boot0

        remaining_time = deadline - time.perf_counter()
        if remaining_time <= 0:
            solver.delete()
            status = "TimeLimitReached"
            break

        t_s0 = time.perf_counter()
        timer = threading.Timer(max(0.01, remaining_time), solver.interrupt)
        timer.start()

        sat = None
        model = None
        try:
            sat = solver.solve_limited(expect_interrupt=True)
            if sat is True:
                model = solver.get_model()
        finally:
            timer.cancel()
            total_solve_time += time.perf_counter() - t_s0
            solver.delete()

        if sat is None or time.perf_counter() >= deadline:
            status = "TimeLimitReached"
            break

        if sat is True:
            perm = encoder.decode_permutation(model)
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

            high = min(mid - 1, actual_bw - 1)
        else:
            low = mid + 1

    if low > high and status != "TimeLimitReached":
        status = "Optimal"

    return {
        "solver": "Pairwise MultipleSAT",
        "status": status,
        "bandwidth": best_bw,
        "permutation": best_perm,
        "encoding_time": total_encoding_time,
        "solve_time": total_solve_time,
    }
