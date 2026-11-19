from __future__ import annotations

import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.order import OrderEncoder
from pysat.solvers import Solver

try:
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def get_rcm_upper_bound(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """Compute initial upper bound and permutation using Reverse Cuthill-McKee heuristic."""
    if not HAS_SCIPY or not edges:
        return n - 1, list(range(n))
    try:
        row = [u for u, v in edges]
        col = [v for u, v in edges]
        data = np.ones(len(edges), dtype=int)
        mat = sp.csr_matrix((data, (row, col)), shape=(n, n))
        mat = mat + mat.T
        perm = csg.reverse_cuthill_mckee(mat)
        inv_perm = np.zeros(n, dtype=int)
        inv_perm[perm] = np.arange(n)
        bw = max(abs(inv_perm[u] - inv_perm[v]) for u, v in edges) if edges else 0
        return int(bw), inv_perm.tolist()
    except Exception:
        return n - 1, list(range(n))


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
    use_2hop: bool = True,
) -> Dict:
    """Incremental SAT solver using Order Encoding with descending threshold search."""
    total_encoding_time = 0.0
    total_solve_time = 0.0
    start_wall_time = time.perf_counter()

    lb = max(1, lower_bound) if lower_bound is not None and lower_bound > 0 else 1

    # Initial upper bound via RCM
    t_rcm0 = time.perf_counter()
    rcm_bw, rcm_perm = get_rcm_upper_bound(n, edges)
    total_encoding_time += (time.perf_counter() - t_rcm0)

    best_bw = rcm_bw
    best_perm = rcm_perm
    status = "Feasible"

    # Early exit if lower bound is already reached
    if best_bw <= lb:
        return {
            "solver": "Order IncrementalSAT",
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    # Base encoding
    t_enc0 = time.perf_counter()
    encoder = OrderEncoder(
        n,
        edges,
        use_strengthening=True,
        use_symmetry_breaking=True,
        use_2hop=use_2hop,
    )
    base_clauses = encoder.get_base_clauses()
    total_encoding_time += (time.perf_counter() - t_enc0)

    # Initialize persistent solver
    solver = Solver(name=solver_name, bootstrap_with=base_clauses)

    k = best_bw - 1

    try:
        while k >= lb:
            elapsed = time.perf_counter() - start_wall_time
            if elapsed >= time_limit:
                status = "TimeLimitReached"
                break

            # Add bandwidth clauses for threshold k
            t_delta0 = time.perf_counter()
            bw_clauses = encoder.get_bandwidth_clauses(k)
            solver.append_formula(bw_clauses)
            total_encoding_time += (time.perf_counter() - t_delta0)

            elapsed = time.perf_counter() - start_wall_time
            if elapsed >= time_limit:
                status = "TimeLimitReached"
                break

            # Solve with timeout guard
            remaining_time = max(0.01, float(time_limit) - elapsed)
            timer = threading.Timer(remaining_time, solver.interrupt)
            timer.start()

            t_s0 = time.perf_counter()
            sat = None
            try:
                sat = solver.solve_limited(expect_interrupt=True)
            finally:
                timer.cancel()
                solver.clear_interrupt()
                s_t = time.perf_counter() - t_s0
                total_solve_time += s_t

            if sat is None or (time.perf_counter() - start_wall_time) >= time_limit:
                status = "TimeLimitReached"
                break

            if sat is True:
                model = solver.get_model()
                perm = encoder.decode_permutation(model)
                actual_bw = max(
                    abs(perm[u - 1] - perm[v - 1])
                    for u, v in encoder.edges
                ) if encoder.edges else 0

                if actual_bw <= best_bw:
                    best_bw = actual_bw
                    best_perm = perm

                if best_bw <= lb:
                    status = "Optimal"
                    break

                # Jump to best_bw - 1
                k = best_bw - 1
            else:
                # UNSAT at k certifies best_bw is optimal
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
    from utils.lb_loader import load_lb_mapping, lookup_lb, get_instance_stem

    test_file = os.path.join(PROJECT_ROOT, "input/small_instances/bcspwr01.mtx.rnd")
    n, edges = load_hb_file(test_file)
    lb_map = load_lb_mapping()
    stem = get_instance_stem(test_file)
    lb = lookup_lb(stem, lb_map)

    print(f"Testing IncrementalSAT on {stem} (n={n}, m={len(edges)}, lb={lb})...")
    res = solve(n, edges, time_limit=60, lower_bound=lb)
    print("Result:", res)
