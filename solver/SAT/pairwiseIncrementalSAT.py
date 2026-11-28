"""Incremental SAT solver isolated for Pairwise Encoding."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

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


def _normalize_zero_indexed_edges(
    n: int, edges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    if n <= 0 or not edges:
        return []
    is_zero = any(u == 0 or v == 0 for u, v in edges)
    if not is_zero:
        max_v = max(max(u, v) for u, v in edges)
        if max_v < n:
            is_zero = True
    norm: List[Tuple[int, int]] = []
    for u, v in edges:
        u0 = u if is_zero else u - 1
        v0 = v if is_zero else v - 1
        if 0 <= u0 < n and 0 <= v0 < n and u0 != v0:
            norm.append((u0, v0))
    return norm


def _multi_rcm_pure_python(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    if n <= 0:
        return 0, []
    if not edges:
        return 0, list(range(n))

    adj = [[] for _ in range(n)]
    for u, v in edges:
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
    if n <= 0:
        return 0, []
    norm_edges = _normalize_zero_indexed_edges(n, edges)
    if not norm_edges:
        return 0, list(range(n))
    best_bw, best_perm = _multi_rcm_pure_python(n, norm_edges)
    if HAS_SCIPY:
        try:
            row = [u for u, v in norm_edges]
            col = [v for u, v in norm_edges]
            data = np.ones(len(norm_edges), dtype=int)
            mat = sp.csr_matrix((data, (row, col)), shape=(n, n))
            mat = mat + mat.T
            perm = csg.reverse_cuthill_mckee(mat)
            inv_perm = np.zeros(n, dtype=int)
            inv_perm[perm] = np.arange(n)
            bw = int(max(abs(inv_perm[u] - inv_perm[v]) for u, v in norm_edges))
            if bw < best_bw:
                best_bw, best_perm = bw, inv_perm.tolist()
        except Exception:
            pass
    return best_bw, best_perm


def _notify_progress(
    cb: Optional[Callable[[int, float, float], None]],
    bw: int,
    enc_t: float,
    sol_t: float,
) -> None:
    if cb is not None:
        try:
            cb(int(bw), float(enc_t), float(sol_t))
        except Exception:
            pass


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 120,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
    search_strategy: str = "binary",
    use_symmetry_breaking: bool = True,
    use_degree_bounds: bool = True,
    use_support: bool = True,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Dict:
    total_encoding_time = 0.0
    total_solve_time = 0.0
    start_wall_time = time.perf_counter()
    deadline = start_wall_time + float(time_limit)
    solver_label = "Pairwise IncrementalSAT"

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

    if time.perf_counter() >= deadline or total_encoding_time >= time_limit:
        return {
            "solver": solver_label,
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
            encoder = PairwiseEncoder(
                n,
                edges,
                use_symmetry_breaking=use_symmetry_breaking,
                use_degree_bounds=use_degree_bounds,
                use_support=use_support,
                deadline=deadline,
            )
            encoder.add_base_clauses(solver, deadline=deadline)
        except (EncodingTimeoutError, MemoryError):
            total_encoding_time += time.perf_counter() - t_enc0
            _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)
            return {
                "solver": solver_label,
                "status": "TimeLimitReached",
                "bandwidth": best_bw,
                "permutation": best_perm,
                "encoding_time": total_encoding_time,
                "solve_time": total_solve_time,
            }
        total_encoding_time += time.perf_counter() - t_enc0
        _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

        if search_strategy == "binary":
            next_act_var = encoder.total_vars + 1
            low = lb
            high = best_bw - 1
            locked_k: Optional[int] = None

            while low <= high:
                if (
                    time.perf_counter() >= deadline
                    or total_encoding_time >= time_limit
                    or total_solve_time >= time_limit
                ):
                    status = "TimeLimitReached"
                    break

                mid = (low + high) // 2
                act_var = next_act_var
                next_act_var += 1

                t_delta0 = time.perf_counter()
                try:
                    feasible = encoder.add_bandwidth_clauses_delta(
                        solver,
                        mid,
                        k_prev=locked_k,
                        act_var=act_var,
                        deadline=deadline,
                    )
                except (EncodingTimeoutError, MemoryError):
                    total_encoding_time += time.perf_counter() - t_delta0
                    status = "TimeLimitReached"
                    break
                total_encoding_time += time.perf_counter() - t_delta0

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

                if sat is None:
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
                    _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

                    if best_bw <= lb:
                        status = "Optimal"
                        break

                    high = min(mid - 1, actual_bw - 1)
                else:
                    solver.add_clause([-act_var])
                    low = mid + 1
                    _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

            if low > high and status != "TimeLimitReached":
                status = "Optimal"
        else:
            k = best_bw - 1
            prev_k: Optional[int] = None
            while k >= lb:
                if (
                    time.perf_counter() >= deadline
                    or total_encoding_time >= time_limit
                    or total_solve_time >= time_limit
                ):
                    status = "TimeLimitReached"
                    break

                t_delta0 = time.perf_counter()
                try:
                    feasible = encoder.add_bandwidth_clauses_delta(
                        solver, k, k_prev=prev_k, deadline=deadline
                    )
                except (EncodingTimeoutError, MemoryError):
                    total_encoding_time += time.perf_counter() - t_delta0
                    status = "TimeLimitReached"
                    break
                prev_k = k
                total_encoding_time += time.perf_counter() - t_delta0

                if not feasible:
                    status = "Optimal"
                    break

                remaining_wall = deadline - time.perf_counter()
                remaining_solve = float(time_limit) - total_solve_time
                remaining_time = min(remaining_wall, remaining_solve)
                if remaining_time <= 0 or total_encoding_time >= time_limit:
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

                if sat is None:
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

                    _notify_progress(progress_callback, best_bw, total_encoding_time, total_solve_time)

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

    print(f"Testing Pairwise IncrementalSAT on {stem} (n={n}, m={len(edges)}, lb={lb})...")
    res = solve(n, edges, time_limit=120, lower_bound=lb)
    print("Result:", res)
