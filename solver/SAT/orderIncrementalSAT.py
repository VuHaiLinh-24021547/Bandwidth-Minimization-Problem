"""Incremental SAT solver for Order Encoding (with assumption-guarded binary search)."""

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

try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def get_rcm_upper_bound(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    if not HAS_SCIPY or not edges:
        return n - 1, list(range(n))
    try:
        row = [u for u, v in edges]
        col = [v for u, v in edges]
        mat = sp.csr_matrix((np.ones(len(edges), dtype=int), (row, col)), shape=(n, n))
        perm = csg.reverse_cuthill_mckee(mat + mat.T)
        inv_perm = np.zeros(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        bw = max(abs(inv_perm[u] - inv_perm[v]) for u, v in edges)
        return int(bw), inv_perm.tolist()
    except Exception:
        return n - 1, list(range(n))


def _identity_upper_bound(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    perm = list(range(n))
    if not edges:
        return 0, perm
    is_zero = any(u == 0 or v == 0 for u, v in edges) or max(max(u, v) for u, v in edges) < n
    bw = max(
        abs(perm[u] - perm[v]) if is_zero else abs(perm[u - 1] - perm[v - 1])
        for u, v in edges
    )
    return bw, perm


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
    search_strategy: str = "binary",
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
            "solver": "Order IncrementalSAT",
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    if time.perf_counter() >= deadline:
        return {
            "solver": "Order IncrementalSAT",
            "status": "TimeLimitReached",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    solver = Solver(name=solver_name)
    try:
        t_enc0 = time.perf_counter()
        try:
            encoder = OrderEncoder(
                n,
                edges,
                use_strengthening=use_strengthening,
                use_symmetry_breaking=use_symmetry_breaking,
                use_degree_bounds=use_degree_bounds,
                use_2hop=use_2hop,
                amo_encoding=amo_encoding,
            )
            encoder.add_base_clauses(solver, deadline=deadline)
        except TimeoutError:
            total_encoding_time += time.perf_counter() - t_enc0
            return {
                "solver": "Order IncrementalSAT",
                "status": "TimeLimitReached",
                "bandwidth": best_bw,
                "permutation": best_perm,
                "encoding_time": total_encoding_time,
                "solve_time": 0.0,
            }
        total_encoding_time += time.perf_counter() - t_enc0

        if search_strategy == "binary":
            next_act_var = encoder.total_vars + 1
            low = lb
            high = best_bw - 1

            while low <= high:
                if time.perf_counter() >= deadline:
                    status = "TimeLimitReached"
                    break

                mid = (low + high) // 2
                act_var = next_act_var
                next_act_var += 1

                t_delta0 = time.perf_counter()
                try:
                    encoder.add_bandwidth_clauses(solver, mid, act_var=act_var, deadline=deadline)
                except TimeoutError:
                    total_encoding_time += time.perf_counter() - t_delta0
                    status = "TimeLimitReached"
                    break
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
                    sat = solver.solve_limited(assumptions=[act_var], expect_interrupt=True)
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
                        abs(perm[u - 1] - perm[v - 1]) for u, v in encoder.edges
                    ) if encoder.edges else 0

                    if actual_bw <= best_bw:
                        best_bw = actual_bw
                        best_perm = perm

                    solver.add_clause([act_var])

                    if best_bw <= lb:
                        status = "Optimal"
                        break

                    high = min(mid - 1, actual_bw - 1)
                else:
                    solver.add_clause([-act_var])
                    low = mid + 1

            if low > high and status != "TimeLimitReached":
                status = "Optimal"
        else:
            k = best_bw - 1
            while k >= lb:
                if time.perf_counter() >= deadline:
                    status = "TimeLimitReached"
                    break

                t_delta0 = time.perf_counter()
                try:
                    encoder.add_bandwidth_clauses(solver, k, deadline=deadline)
                except TimeoutError:
                    total_encoding_time += time.perf_counter() - t_delta0
                    status = "TimeLimitReached"
                    break
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
                        abs(perm[u - 1] - perm[v - 1]) for u, v in encoder.edges
                    ) if encoder.edges else 0

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
        "solver": "Order IncrementalSAT",
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
