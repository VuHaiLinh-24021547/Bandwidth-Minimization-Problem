from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.pairwise import EncodingTimeoutError, PairwiseEncoder
from pysat.solvers import Solver

try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _multi_rcm_pure_python(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    if not edges:
        return 0, list(range(n))

    adj = [[] for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].append(v)
            adj[v].append(u)

    deg = [len(adj[i]) for i in range(n)]
    for i in range(n):
        adj[i].sort(key=lambda x: deg[x])

    def bfs_order(starts: List[int]) -> Tuple[int, List[int]]:
        visited = [False] * n
        order: List[int] = []
        for s in starts:
            if visited[s]:
                continue
            visited[s] = True
            q = deque([s])
            comp = []
            while q:
                u = q.popleft()
                comp.append(u)
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        q.append(w)
            comp.reverse()
            order.extend(comp)
        inv_perm = [0] * n
        for pos, node in enumerate(order):
            inv_perm[node] = pos
        bw = max(abs(inv_perm[u] - inv_perm[v]) for u, v in edges)
        return int(bw), inv_perm

    nodes_by_deg = sorted(range(n), key=lambda x: (deg[x], x))
    best_bw, best_perm = bfs_order(nodes_by_deg)

    if n > 35:
        candidates = list(nodes_by_deg[:min(n, 20)])
        for s0 in nodes_by_deg[:min(n, 5)]:
            dist = [-1] * n
            dist[s0] = 0
            q = deque([s0])
            last = s0
            while q:
                u = q.popleft()
                last = u
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        q.append(w)
            if last not in candidates:
                candidates.append(last)

        for s in candidates:
            bw, perm = bfs_order([s] + nodes_by_deg)
            if bw < best_bw:
                best_bw, best_perm = bw, perm

    return best_bw, best_perm


def get_rcm_upper_bound(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    if not edges:
        return 0, list(range(n))
    best_bw, best_perm = _multi_rcm_pure_python(n, edges)
    if HAS_SCIPY:
        try:
            row = [u for u, v in edges]
            col = [v for u, v in edges]
            data = np.ones(len(edges), dtype=int)
            mat = sp.csr_matrix((data, (row, col)), shape=(n, n))
            mat = mat + mat.T
            perm = csg.reverse_cuthill_mckee(mat)
            inv_perm = np.zeros(n, dtype=int)
            inv_perm[perm] = np.arange(n)
            bw = int(max(abs(inv_perm[u] - inv_perm[v]) for u, v in edges))
            if bw < best_bw:
                best_bw, best_perm = bw, inv_perm.tolist()
        except Exception:
            pass
    return best_bw, best_perm


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
    search_strategy: str = "binary",
) -> Dict:
    total_encoding_time = 0.0
    total_solve_time = 0.0
    start_wall_time = time.perf_counter()
    deadline = start_wall_time + float(time_limit)
    solver_label = "Pairwise IncrementalSAT"

    lb = max(1, lower_bound) if lower_bound is not None and lower_bound > 0 else 1

    t_rcm0 = time.perf_counter()
    rcm_bw, rcm_perm = get_rcm_upper_bound(n, edges)
    total_encoding_time += (time.perf_counter() - t_rcm0)

    best_bw = rcm_bw
    best_perm = rcm_perm
    status = "Feasible"

    if best_bw <= lb:
        return {
            "solver": solver_label,
            "status": "Optimal",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": 0.0,
        }

    t_enc0 = time.perf_counter()
    try:
        encoder = PairwiseEncoder(
            n,
            edges,
            use_symmetry_breaking=True,
            use_degree_bounds=True,
            use_support=True,
        )
        base_clauses = encoder.get_base_clauses(deadline=deadline)
    except EncodingTimeoutError:
        total_encoding_time += (time.perf_counter() - t_enc0)
        return {
            "solver": solver_label,
            "status": "TimeLimitReached",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": total_solve_time,
        }
    total_encoding_time += (time.perf_counter() - t_enc0)

    if time.perf_counter() >= deadline:
        return {
            "solver": solver_label,
            "status": "TimeLimitReached",
            "bandwidth": best_bw,
            "permutation": best_perm,
            "encoding_time": total_encoding_time,
            "solve_time": total_solve_time,
        }

    solver = Solver(name=solver_name, bootstrap_with=base_clauses)

    try:
        if search_strategy == "binary":
            next_act_var = encoder.total_vars + 1
            low = lb
            high = best_bw - 1
            locked_k: Optional[int] = None

            while low <= high:
                elapsed = time.perf_counter() - start_wall_time
                if elapsed >= time_limit:
                    status = "TimeLimitReached"
                    break

                mid = (low + high) // 2
                act_var = next_act_var
                next_act_var += 1

                t_delta0 = time.perf_counter()
                try:
                    bw_clauses = encoder.get_bandwidth_clauses_delta(
                        mid,
                        k_prev=locked_k,
                        act_var=act_var,
                        deadline=deadline,
                    )
                except EncodingTimeoutError:
                    total_encoding_time += (time.perf_counter() - t_delta0)
                    status = "TimeLimitReached"
                    break
                solver.append_formula(bw_clauses)
                total_encoding_time += (time.perf_counter() - t_delta0)

                elapsed = time.perf_counter() - start_wall_time
                if elapsed >= time_limit:
                    status = "TimeLimitReached"
                    break

                remaining_time = max(0.01, float(time_limit) - elapsed)
                timer = threading.Timer(remaining_time, solver.interrupt)
                timer.start()

                t_s0 = time.perf_counter()
                sat = None
                try:
                    sat = solver.solve_limited(assumptions=[act_var], expect_interrupt=True)
                finally:
                    timer.cancel()
                    solver.clear_interrupt()
                    total_solve_time += (time.perf_counter() - t_s0)

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

                    solver.add_clause([act_var])
                    locked_k = mid

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
            prev_k: Optional[int] = None
            while k >= lb:
                elapsed = time.perf_counter() - start_wall_time
                if elapsed >= time_limit:
                    status = "TimeLimitReached"
                    break

                t_delta0 = time.perf_counter()
                try:
                    bw_clauses = encoder.get_bandwidth_clauses_delta(k, prev_k, deadline=deadline)
                except EncodingTimeoutError:
                    total_encoding_time += (time.perf_counter() - t_delta0)
                    status = "TimeLimitReached"
                    break
                solver.append_formula(bw_clauses)
                prev_k = k
                total_encoding_time += (time.perf_counter() - t_delta0)

                elapsed = time.perf_counter() - start_wall_time
                if elapsed >= time_limit:
                    status = "TimeLimitReached"
                    break

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
                    total_solve_time += (time.perf_counter() - t_s0)

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

                    k = best_bw - 1
                else:
                    status = "Optimal"
                    break

            if k < lb and status != "TimeLimitReached":
                status = "Optimal"

    finally:
        solver.delete()

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

    print(f"Testing Pairwise IncrementalSAT on {stem} (n={n}, m={len(edges)}, lb={lb})...")
    res = solve(n, edges, time_limit=60, lower_bound=lb)
    print("Result:", res)
