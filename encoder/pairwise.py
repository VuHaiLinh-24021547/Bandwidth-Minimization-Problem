from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class EncodingTimeoutError(TimeoutError):
    pass


class PairwiseEncoder:
    def __init__(
        self,
        n: int,
        edges: List[Tuple[int, int]],
        use_symmetry_breaking: bool = True,
        use_degree_bounds: bool = True,
        use_support: bool = True,
        deadline: Optional[float] = None,
    ):
        self.n = n
        self.use_symmetry_breaking = use_symmetry_breaking
        self.use_degree_bounds = use_degree_bounds
        self.use_support = use_support

        is_zero_indexed = any(u == 0 or v == 0 for u, v in edges) if edges else False
        if not is_zero_indexed and edges:
            max_v = max(max(u, v) for u, v in edges)
            if max_v < n:
                is_zero_indexed = True

        self.is_zero_indexed_input = is_zero_indexed

        norm_edges = set()
        self.degrees = [0] * (n + 1)
        adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            u_norm = u + 1 if is_zero_indexed else u
            v_norm = v + 1 if is_zero_indexed else v
            if u_norm != v_norm and 1 <= u_norm <= n and 1 <= v_norm <= n:
                edge = (min(u_norm, v_norm), max(u_norm, v_norm))
                if edge not in norm_edges:
                    norm_edges.add(edge)
                    self.degrees[u_norm] += 1
                    self.degrees[v_norm] += 1
                    adj[u_norm].append(v_norm)
                    adj[v_norm].append(u_norm)

        self.edges: List[Tuple[int, int]] = sorted(norm_edges)
        self.adj = adj
        self.anchor_vertex = (
            max(range(1, n + 1), key=lambda v: (self.degrees[v], -v))
            if n > 0
            else 1
        )
        self.total_vars = n * n
        self._base_clauses: Optional[List[List[int]]] = None

        self.dist, self.balls = self._precompute_bfs(deadline=deadline)

    def _precompute_bfs(
        self, deadline: Optional[float] = None
    ) -> Tuple[List[List[int]], List[List[int]]]:
        n = self.n
        adj = self.adj
        dist = [[-1] * (n + 1) for _ in range(n + 1)]
        balls = [[0] * (n + 1) for _ in range(n + 1)]
        for s in range(1, n + 1):
            if (s & 15) == 0:
                self._check_deadline(deadline)
            d = dist[s]
            d[s] = 0
            q = deque([s])
            cnt = [0] * (n + 1)
            while q:
                u = q.popleft()
                du = d[u]
                if du > 0:
                    cnt[du] += 1
                nd = du + 1
                for w in adj[u]:
                    if d[w] == -1:
                        d[w] = nd
                        q.append(w)
            acc = 0
            bs = balls[s]
            for h in range(1, n + 1):
                acc += cnt[h]
                bs[h] = acc
        return dist, balls

    def var_X(self, v: int, a: int) -> int:
        if not (1 <= v <= self.n and 1 <= a <= self.n):
            raise ValueError(f"Invalid indices for X(v={v}, a={a}) with n={self.n}")
        return (v - 1) * self.n + a

    def _check_deadline(self, deadline: Optional[float]) -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise EncodingTimeoutError("Encoding time limit exceeded")

    def compute_domain_bounds(self, k: int) -> Tuple[List[int], List[int], bool]:
        n = self.n
        min_p = [1] * (n + 1)
        max_p = [n] * (n + 1)

        if self.use_symmetry_breaking and n >= 2:
            max_p[self.anchor_vertex] = (n + 1) // 2

        if k >= n - 1:
            return min_p, max_p, True

        if self.use_degree_bounds:
            anchor = self.anchor_vertex
            d_anchor = self.dist[anchor]
            anchor_max = max_p[anchor]
            for u in range(1, n + 1):
                bu = self.balls[u]
                for h in range(1, n):
                    hk = h * k
                    if hk >= n - 1:
                        break
                    c = bu[h]
                    if c > hk:
                        lo = c - hk + 1
                        hi = n - c + hk
                        if lo > min_p[u]:
                            min_p[u] = lo
                        if hi < max_p[u]:
                            max_p[u] = hi
                if self.use_symmetry_breaking and n >= 2:
                    da = d_anchor[u]
                    if da > 0:
                        hi_a = anchor_max + da * k
                        if hi_a < max_p[u]:
                            max_p[u] = hi_a

            changed = True
            while changed:
                changed = False
                for u, v in self.edges:
                    if min_p[v] - k > min_p[u]:
                        min_p[u] = min_p[v] - k
                        changed = True
                    if max_p[v] + k < max_p[u]:
                        max_p[u] = max_p[v] + k
                        changed = True
                    if min_p[u] - k > min_p[v]:
                        min_p[v] = min_p[u] - k
                        changed = True
                    if max_p[u] + k < max_p[v]:
                        max_p[v] = max_p[u] + k
                        changed = True

            for u in range(1, n + 1):
                if min_p[u] > max_p[u]:
                    return min_p, max_p, False

        return min_p, max_p, True

    def get_base_clauses(self, deadline: Optional[float] = None) -> List[List[int]]:
        if self._base_clauses is not None:
            return self._base_clauses

        clauses: List[List[int]] = []
        n = self.n

        for v in range(1, n + 1):
            base = (v - 1) * n
            clauses.append([base + a for a in range(1, n + 1)])

        for a in range(1, n + 1):
            clauses.append([(v - 1) * n + a for v in range(1, n + 1)])

        for v in range(1, n + 1):
            self._check_deadline(deadline)
            base = (v - 1) * n
            for a in range(1, n):
                xa = -(base + a)
                for b in range(a + 1, n + 1):
                    clauses.append([xa, -(base + b)])

        for a in range(1, n + 1):
            self._check_deadline(deadline)
            for u in range(1, n):
                xu = -((u - 1) * n + a)
                for v in range(u + 1, n + 1):
                    clauses.append([xu, -((v - 1) * n + a)])

        if self.use_symmetry_breaking and n >= 2:
            mid_cutoff = (n + 1) // 2 + 1
            base = (self.anchor_vertex - 1) * n
            for a in range(mid_cutoff, n + 1):
                clauses.append([-(base + a)])

        self._base_clauses = clauses
        return clauses

    def get_bandwidth_clauses(
        self,
        k: int,
        act_var: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> List[List[int]]:
        clauses: List[List[int]] = []
        n = self.n

        if k >= n - 1:
            return clauses

        min_p, max_p, feasible = self.compute_domain_bounds(k)
        prefix = [-act_var] if act_var is not None else []

        if not feasible:
            b = 1
            return [prefix + [b], prefix + [-b]]

        for u in range(1, n + 1):
            base = (u - 1) * n
            for a in range(1, min_p[u]):
                clauses.append(prefix + [-(base + a)])
            for a in range(max_p[u] + 1, n + 1):
                clauses.append(prefix + [-(base + a)])

        for u, v in self.edges:
            self._check_deadline(deadline)
            base_u = (u - 1) * n
            base_v = (v - 1) * n
            mpu, xpu = min_p[u], max_p[u]
            mpv, xpv = min_p[v], max_p[v]

            for a in range(mpu, xpu + 1):
                xua = -(base_u + a)
                hi_left = min(xpv + 1, a - k)
                for b in range(mpv, hi_left):
                    clauses.append(prefix + [xua, -(base_v + b)])
                lo_right = max(mpv, a + k + 1)
                for b in range(lo_right, xpv + 1):
                    clauses.append(prefix + [xua, -(base_v + b)])
                if self.use_support:
                    lo = max(mpv, a - k)
                    hi = min(xpv, a + k)
                    if lo > hi:
                        clauses.append(prefix + [xua])
                    elif lo > mpv or hi < xpv:
                        clauses.append(prefix + [xua] + [base_v + b for b in range(lo, hi + 1)])

            if self.use_support:
                for b in range(mpv, xpv + 1):
                    xvb = -(base_v + b)
                    lo = max(mpu, b - k)
                    hi = min(xpu, b + k)
                    if lo > hi:
                        clauses.append(prefix + [xvb])
                    elif lo > mpu or hi < xpu:
                        clauses.append(prefix + [xvb] + [base_u + a for a in range(lo, hi + 1)])

        return clauses

    def get_bandwidth_clauses_delta(
        self,
        k_new: int,
        k_prev: Optional[int] = None,
        act_var: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> List[List[int]]:
        n = self.n
        if k_prev is None or k_prev >= n - 1:
            return self.get_bandwidth_clauses(k_new, act_var=act_var, deadline=deadline)

        clauses: List[List[int]] = []
        if k_new >= k_prev:
            return clauses

        min_prev, max_prev, _ = self.compute_domain_bounds(k_prev)
        min_p, max_p, feasible = self.compute_domain_bounds(k_new)
        prefix = [-act_var] if act_var is not None else []

        if not feasible:
            b = 1
            return [prefix + [b], prefix + [-b]]

        for u in range(1, n + 1):
            base = (u - 1) * n
            for a in range(min_prev[u], min(min_p[u], n + 1)):
                clauses.append(prefix + [-(base + a)])
            for a in range(max(1, max_p[u] + 1), max_prev[u] + 1):
                clauses.append(prefix + [-(base + a)])

        upper_dist = min(k_prev, n - 1)
        for u, v in self.edges:
            self._check_deadline(deadline)
            base_u = (u - 1) * n
            base_v = (v - 1) * n
            mpu, xpu = min_p[u], max_p[u]
            mpv, xpv = min_p[v], max_p[v]

            for dist in range(k_new + 1, upper_dist + 1):
                for a in range(mpu, min(xpu, n - dist) + 1):
                    b = a + dist
                    if mpv <= b <= xpv:
                        clauses.append(prefix + [-(base_u + a), -(base_v + b)])
                for b in range(mpv, min(xpv, n - dist) + 1):
                    a = b + dist
                    if mpu <= a <= xpu:
                        clauses.append(prefix + [-(base_v + b), -(base_u + a)])

            if self.use_support:
                for a in range(mpu, xpu + 1):
                    xua = -(base_u + a)
                    lo = max(mpv, a - k_new)
                    hi = min(xpv, a + k_new)
                    if lo > hi:
                        clauses.append(prefix + [xua])
                    elif lo > mpv or hi < xpv:
                        clauses.append(prefix + [xua] + [base_v + b for b in range(lo, hi + 1)])
                for b in range(mpv, xpv + 1):
                    xvb = -(base_v + b)
                    lo = max(mpu, b - k_new)
                    hi = min(xpu, b + k_new)
                    if lo > hi:
                        clauses.append(prefix + [xvb])
                    elif lo > mpu or hi < xpu:
                        clauses.append(prefix + [xvb] + [base_u + a for a in range(lo, hi + 1)])

        return clauses

    def encode(self, k: int, deadline: Optional[float] = None) -> List[List[int]]:
        n = self.n
        if k >= n - 1:
            return self.get_base_clauses(deadline=deadline)

        min_p, max_p, feasible = self.compute_domain_bounds(k)
        if not feasible:
            return [[1], [-1]]

        clauses: List[List[int]] = []

        for u in range(1, n + 1):
            self._check_deadline(deadline)
            base = (u - 1) * n
            mpu, xpu = min_p[u], max_p[u]
            clauses.append([base + a for a in range(mpu, xpu + 1)])
            for a in range(1, mpu):
                clauses.append([-(base + a)])
            for a in range(xpu + 1, n + 1):
                clauses.append([-(base + a)])
            for a in range(mpu, xpu):
                xa = -(base + a)
                for b in range(a + 1, xpu + 1):
                    clauses.append([xa, -(base + b)])

        for a in range(1, n + 1):
            self._check_deadline(deadline)
            valid_u = [u for u in range(1, n + 1) if min_p[u] <= a <= max_p[u]]
            if not valid_u:
                return [[1], [-1]]
            clauses.append([(u - 1) * n + a for u in valid_u])
            for i in range(len(valid_u) - 1):
                xua = -((valid_u[i] - 1) * n + a)
                for j in range(i + 1, len(valid_u)):
                    clauses.append([xua, -((valid_u[j] - 1) * n + a)])

        for u, v in self.edges:
            self._check_deadline(deadline)
            base_u = (u - 1) * n
            base_v = (v - 1) * n
            mpu, xpu = min_p[u], max_p[u]
            mpv, xpv = min_p[v], max_p[v]

            for a in range(mpu, xpu + 1):
                xua = -(base_u + a)
                hi_left = min(xpv + 1, a - k)
                for b in range(mpv, hi_left):
                    clauses.append([xua, -(base_v + b)])
                lo_right = max(mpv, a + k + 1)
                for b in range(lo_right, xpv + 1):
                    clauses.append([xua, -(base_v + b)])
                if self.use_support:
                    lo = max(mpv, a - k)
                    hi = min(xpv, a + k)
                    if lo > hi:
                        clauses.append([xua])
                    elif lo > mpv or hi < xpv:
                        clauses.append([xua] + [base_v + b for b in range(lo, hi + 1)])

            if self.use_support:
                for b in range(mpv, xpv + 1):
                    xvb = -(base_v + b)
                    lo = max(mpu, b - k)
                    hi = min(xpu, b + k)
                    if lo > hi:
                        clauses.append([xvb])
                    elif lo > mpu or hi < xpu:
                        clauses.append([xvb] + [base_u + a for a in range(lo, hi + 1)])

        return clauses

    def decode(self, model: List[int]) -> Dict[int, int]:
        model_set: Set[int] = set(model)
        labeling: Dict[int, int] = {}
        n = self.n
        for v in range(1, n + 1):
            base = (v - 1) * n
            for a in range(1, n + 1):
                if (base + a) in model_set:
                    labeling[v] = a
                    break
        return labeling

    def decode_permutation(self, model: List[int]) -> List[int]:
        labeling = self.decode(model)
        perm = [0] * self.n
        for v in range(1, self.n + 1):
            perm[v - 1] = labeling[v] - 1
        return perm

    def compute_bandwidth(self, labeling: Dict[int, int]) -> int:
        if not self.edges:
            return 0
        return max(abs(labeling[u] - labeling[v]) for u, v in self.edges)

    def is_valid_labeling(self, labeling: Dict[int, int]) -> bool:
        if len(labeling) != self.n:
            return False
        return set(labeling.values()) == set(range(1, self.n + 1))

    def to_dimacs(self, clauses: List[List[int]]) -> str:
        lines = [f"p cnf {self.total_vars} {len(clauses)}"]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)
