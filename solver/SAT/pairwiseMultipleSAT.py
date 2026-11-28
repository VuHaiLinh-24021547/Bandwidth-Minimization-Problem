"""Multiple SAT solver isolated for Pairwise Encoding."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.pairwise import EncodingTimeoutError, PairwiseEncoder
from pysat.solvers import Solver
from solver.SAT.pairwiseIncrementalSAT import _notify_progress, get_rcm_upper_bound


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 120,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
    use_symmetry_breaking: bool = True,
    use_degree_bounds: bool = True,
    use_support: bool = True,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Dict:
    total_encoding_time = 0.0
    total_solve_time = 0.0
    start_wall_time = time.perf_counter()
    deadline = start_wall_time + float(time_limit)
    solver_label = "Pairwise MultipleSAT"

    lb = max(1, lower_bound) if lower_bound is not None and lower_bound > 0 else 1

    t_rcm0 = time.perf_counter()
    rcm_bw, rcm_perm = get_rcm_upper_bound(n, edges)
    total_encoding_time += time.perf_counter() - t_rcm0

    best_bw = rcm_bw
    best_perm = rcm_perm
    status = "Feasible"
    _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

    if best_bw <= lb or not edges or n <= 1:
        return {
            "solver": solver_label,
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    t_init0 = time.perf_counter()
    try:
        encoder = PairwiseEncoder(
            n,
            edges,
            use_symmetry_breaking=use_symmetry_breaking,
            use_degree_bounds=use_degree_bounds,
            use_support=use_support,
            deadline=deadline,
        )
    except (EncodingTimeoutError, MemoryError):
        total_encoding_time += time.perf_counter() - t_init0
        _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)
        return {
            "solver": solver_label,
            "status": "TimeLimitReached",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": total_solve_time,
        }
    total_encoding_time += time.perf_counter() - t_init0

    low = lb
    high = best_bw - 1

    while low <= high:
        if (
            time.perf_counter() >= deadline
            or total_encoding_time >= time_limit
            or total_solve_time >= time_limit
        ):
            status = "TimeLimitReached"
            break

        mid = (low + high) // 2

        solver = Solver(name=solver_name)
        sat = None
        model = None
        try:
            t_enc0 = time.perf_counter()
            try:
                feasible = encoder.add_encoded_clauses(solver, mid, deadline=deadline)
            except (EncodingTimeoutError, MemoryError):
                total_encoding_time += time.perf_counter() - t_enc0
                status = "TimeLimitReached"
                break
            total_encoding_time += time.perf_counter() - t_enc0

            if not feasible:
                low = mid + 1
                _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)
                continue

            remaining_wall = deadline - time.perf_counter()
            remaining_solve = float(time_limit) - total_solve_time
            remaining_time = min(remaining_wall, remaining_solve)
            if remaining_time <= 0 or total_encoding_time >= time_limit:
                status = "TimeLimitReached"
                break

            t_s0 = time.perf_counter()
            timer = threading.Timer(max(0.01, remaining_time), solver.interrupt)
            timer.start()
            try:
                sat = solver.solve_limited(expect_interrupt=True)
                if sat is True:
                    model = solver.get_model()
            finally:
                timer.cancel()
                total_solve_time += time.perf_counter() - t_s0
        finally:
            solver.delete()

        if sat is None:
            status = "TimeLimitReached"
            break

        if sat is True:
            perm = encoder.decode_permutation(model)
            actual_bw = max(
                abs(perm[u - 1] - perm[v - 1])
                for u, v in encoder.edges
            ) if encoder.edges else 0

            if actual_bw <= best_bw:
                best_bw = actual_bw
                best_perm = perm

            _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

            if best_bw <= lb:
                status = "Optimal"
                break

            high = min(mid - 1, actual_bw - 1)
        else:
            low = mid + 1
            _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

    if low > high and status != "TimeLimitReached":
        status = "Optimal"

    _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)
    return {
        "solver": solver_label,
        "status": status,
        "bandwidth": best_bw,
        "permutation": best_perm,
        "encoding_time": total_encoding_time,
        "solve_time": total_solve_time,
    }


solve_pairwise = solve


if __name__ == "__main__":
    from utils.hb_parser import load_hb_file
    from utils.lb_loader import get_instance_stem, load_lb_mapping, lookup_lb

    test_file = os.path.join(PROJECT_ROOT, "input/small_instances/bcspwr01.mtx.rnd")
    n, edges = load_hb_file(test_file)
    lb_map = load_lb_mapping()
    stem = get_instance_stem(test_file)
    lb = lookup_lb(stem, lb_map)

    print(f"Testing Pairwise MultipleSAT on {stem} (n={n}, m={len(edges)}, lb={lb})...")
    res = solve(n, edges, time_limit=120, lower_bound=lb)
    print("Result:", res)
