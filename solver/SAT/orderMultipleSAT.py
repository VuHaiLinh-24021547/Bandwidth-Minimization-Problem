"""Multiple SAT solver for Order Encoding (binary search over independent SAT instances)."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.order import OrderEncoder
from solver.SAT.orderIncrementalSAT import _identity_upper_bound, get_rcm_upper_bound


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
    use_rcm: bool = True,
    use_strengthening: bool = True,
    use_symmetry_breaking: bool = True,
    use_degree_bounds: bool = True,
    use_2hop: bool = True,
    amo_encoding: str = "seqcounter",
) -> Dict:
    from pysat.solvers import Solver

    total_encoding_time = 0.0
    total_solve_time = 0.0
    start_wall_time = time.perf_counter()
    deadline = start_wall_time + float(time_limit)

    lb = max(1, lower_bound) if lower_bound is not None and lower_bound > 0 else 1

    t0 = time.perf_counter()
    best_bw, best_perm = get_rcm_upper_bound(n, edges) if use_rcm else _identity_upper_bound(n, edges)
    total_encoding_time += time.perf_counter() - t0
    status = "Feasible"

    if best_bw <= lb:
        return {
            "solver": "Order MultipleSAT",
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    if time.perf_counter() >= deadline:
        return {
            "solver": "Order MultipleSAT",
            "status": "TimeLimitReached",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    t_init0 = time.perf_counter()
    encoder = OrderEncoder(
        n,
        edges,
        use_strengthening=use_strengthening,
        use_symmetry_breaking=use_symmetry_breaking,
        use_degree_bounds=use_degree_bounds,
        use_2hop=use_2hop,
        amo_encoding=amo_encoding,
    )
    total_encoding_time += time.perf_counter() - t_init0

    low = lb
    high = best_bw - 1

    while low <= high:
        if time.perf_counter() >= deadline:
            status = "TimeLimitReached"
            break

        mid = (low + high) // 2

        solver = Solver(name=solver_name)
        sat = None
        model = None
        try:
            t_enc0 = time.perf_counter()
            try:
                encoder.add_base_clauses(solver, deadline=deadline)
                encoder.add_bandwidth_clauses(solver, mid, deadline=deadline)
            except TimeoutError:
                total_encoding_time += time.perf_counter() - t_enc0
                status = "TimeLimitReached"
                break
            total_encoding_time += time.perf_counter() - t_enc0

            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
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

        if sat is None or time.perf_counter() >= deadline:
            status = "TimeLimitReached"
            break

        if sat is True:
            perm = encoder.decode_permutation(model)
            actual_bw = max(
                abs(perm[u - 1] - perm[v - 1]) for u, v in encoder.edges
            ) if encoder.edges else 0

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
        "solver": "Order MultipleSAT",
        "status": status,
        "bandwidth": best_bw,
        "permutation": best_perm,
        "encoding_time": total_encoding_time,
        "solve_time": total_solve_time,
    }


if __name__ == "__main__":
    from utils.hb_parser import load_hb_file
    from utils.lb_loader import get_instance_stem, load_lb_mapping, lookup_lb

    test_file = os.path.join(PROJECT_ROOT, "input/small_instances/bcspwr01.mtx.rnd")
    n, edges = load_hb_file(test_file)
    stem = get_instance_stem(test_file)
    lb = lookup_lb(stem, load_lb_mapping())
    print(solve(n, edges, time_limit=60, lower_bound=lb))
