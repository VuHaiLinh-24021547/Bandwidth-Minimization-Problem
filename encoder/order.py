"""Order & Hybrid CNF Encoding for the Bandwidth Minimization Problem."""

from typing import Callable, Dict, List, Optional, Set, Tuple
import time


class OrderEncoder:
    def __init__(
        self,
        n: int,
        edges: List[Tuple[int, int]],
        use_strengthening: bool = True,
        use_symmetry_breaking: bool = True,
        use_degree_bounds: bool = True,
        use_2hop: bool = True,
        amo_encoding: str = "seqcounter",
    ):
        self.n = n
        self.use_strengthening = use_strengthening
        self.use_symmetry_breaking = use_symmetry_breaking
        self.use_degree_bounds = use_degree_bounds
        self.use_2hop = use_2hop
        self.amo_encoding = amo_encoding.lower()

        is_zero_indexed = any(u == 0 or v == 0 for u, v in edges) if edges else False
        if not is_zero_indexed and edges:
            if max(max(u, v) for u, v in edges) < n:
                is_zero_indexed = True
        self.is_zero_indexed_input = is_zero_indexed

        norm_edges = set()
        self.degrees = [0] * (n + 1)
        for u, v in edges:
            u_norm = u + 1 if is_zero_indexed else u
            v_norm = v + 1 if is_zero_indexed else v
            if u_norm != v_norm and 1 <= u_norm <= n and 1 <= v_norm <= n:
                edge = (min(u_norm, v_norm), max(u_norm, v_norm))
                if edge not in norm_edges:
                    norm_edges.add(edge)
                    self.degrees[u_norm] += 1
                    self.degrees[v_norm] += 1

        self.edges: List[Tuple[int, int]] = sorted(norm_edges)

        adj = [set() for _ in range(n + 1)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        self.adj = adj

        two_hop_set = set()
        if self.use_2hop:
            for u in range(1, n + 1):
                for w in adj[u]:
                    for v in adj[w]:
                        if v > u and v not in adj[u]:
                            two_hop_set.add((u, v))
        self.two_hop_edges: List[Tuple[int, int]] = sorted(two_hop_set)

        self.anchor_vertex = max(range(1, n + 1), key=lambda v: (self.degrees[v], -v))

        self.num_order_vars = n * (n - 1)
        self.num_exact_vars = n * n
        self.num_aux_vars = n * (n - 1) if self.amo_encoding == "seqcounter" else 0
        self.total_vars = self.num_order_vars + self.num_exact_vars + self.num_aux_vars
        self._base_clauses: Optional[List[List[int]]] = None

    def var_O(self, v: int, t: int) -> int:
        """O(v, t) <=> f(v) >= t (v in 1..n, t in 2..n)."""
        if not (1 <= v <= self.n and 2 <= t <= self.n):
            raise ValueError(f"Invalid indices for O(v={v}, t={t}) with n={self.n}")
        return (v - 1) * (self.n - 1) + (t - 1)

    def var_X(self, v: int, a: int) -> int:
        """X(v, a) <=> f(v) == a (v in 1..n, a in 1..n)."""
        if not (1 <= v <= self.n and 1 <= a <= self.n):
            raise ValueError(f"Invalid indices for X(v={v}, a={a}) with n={self.n}")
        return self.num_order_vars + (v - 1) * self.n + a

    def var_S(self, i: int, a: int) -> int:
        """Sequential counter register S(i, a) (i in 1..n-1, a in 1..n)."""
        if not (1 <= i <= self.n - 1 and 1 <= a <= self.n):
            raise ValueError(f"Invalid indices for S(i={i}, a={a}) with n={self.n}")
        return self.num_order_vars + self.num_exact_vars + (a - 1) * (self.n - 1) + i

    def _emit_base_clauses_chunked(
        self,
        flush_chunk: Callable[[List[List[int]]], None],
        deadline: Optional[float] = None,
        chunk_size: int = 32768,
    ) -> None:
        n = self.n
        nm1 = n - 1
        num_o = self.num_order_vars
        s_offset = num_o + self.num_exact_vars
        buf: List[List[int]] = []
        append = buf.append

        def maybe_flush() -> None:
            if len(buf) >= chunk_size:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("Encoding time limit exceeded")
                flush_chunk(buf)
                buf.clear()

        # Monotonicity: O(v, t+1) => O(v, t)
        for v in range(1, n + 1):
            o_base = (v - 1) * nm1 - 1
            for t in range(2, n):
                append([-(o_base + t + 1), o_base + t])
            maybe_flush()

        # Channeling: X(v, a) <=> O(v, a) and not O(v, a+1)
        for v in range(1, n + 1):
            o_base = (v - 1) * nm1 - 1
            x_base = num_o + (v - 1) * n
            append([-(x_base + 1), -(o_base + 2)])
            append([x_base + 1, o_base + 2])

            for a in range(2, n):
                xa = x_base + a
                oa = o_base + a
                oa1 = oa + 1
                append([-xa, oa])
                append([-xa, -oa1])
                append([-oa, oa1, xa])

            xn = x_base + n
            on = o_base + n
            append([-xn, on])
            append([xn, -on])
            maybe_flush()

        # Bijectivity (ALO on vertices and positions)
        for v in range(1, n + 1):
            x_base = num_o + (v - 1) * n
            append([x_base + a for a in range(1, n + 1)])
        maybe_flush()

        for a in range(1, n + 1):
            append([num_o + (v - 1) * n + a for v in range(1, n + 1)])
        maybe_flush()

        # AMO on positions (Sinz Sequential Counter O(n^2) vs Pairwise O(n^3))
        if self.amo_encoding == "seqcounter":
            for a in range(1, n + 1):
                s_base = s_offset + (a - 1) * nm1
                append([-(num_o + a), s_base + 1])
                for i in range(2, n):
                    xi = num_o + (i - 1) * n + a
                    si = s_base + i
                    sim1 = si - 1
                    append([-xi, si])
                    append([-sim1, si])
                    append([-sim1, -xi])
                append([-(s_base + nm1), -(num_o + (n - 1) * n + a)])
                maybe_flush()
        else:
            for a in range(1, n + 1):
                for u in range(1, n + 1):
                    xu = num_o + (u - 1) * n + a
                    for v in range(u + 1, n + 1):
                        append([-xu, -(num_o + (v - 1) * n + a)])
                    maybe_flush()

        # Reflection symmetry breaking on max-degree anchor vertex
        if self.use_symmetry_breaking and n >= 2:
            mid_cutoff = (n + 1) // 2 + 1
            if mid_cutoff <= n:
                append([-((self.anchor_vertex - 1) * nm1 + (mid_cutoff - 1))])

        if buf:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Encoding time limit exceeded")
            flush_chunk(buf)
            buf.clear()

    def _emit_bandwidth_clauses_chunked(
        self,
        k: int,
        flush_chunk: Callable[[List[List[int]]], None],
        act_var: Optional[int] = None,
        deadline: Optional[float] = None,
        chunk_size: int = 32768,
    ) -> None:
        n = self.n
        if k >= n - 1:
            return

        nm1 = n - 1
        num_o = self.num_order_vars
        buf: List[List[int]] = []
        append = buf.append
        has_act = act_var is not None
        neg_act = -act_var if has_act else 0

        def maybe_flush() -> None:
            if len(buf) >= chunk_size:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("Encoding time limit exceeded")
                flush_chunk(buf)
                buf.clear()

        if self.use_degree_bounds:
            for u in range(1, n + 1):
                d = self.degrees[u]
                if d > k:
                    o_base_u = (u - 1) * nm1 - 1
                    min_pos = d - k + 1
                    if 2 <= min_pos <= n:
                        append([neg_act, o_base_u + min_pos] if has_act else [o_base_u + min_pos])
                    max_pos = n - d + k
                    if 1 <= max_pos <= n - 1:
                        lit = -(o_base_u + max_pos + 1)
                        append([neg_act, lit] if has_act else [lit])
            maybe_flush()

        for u, v in self.edges:
            o_base_u = (u - 1) * nm1 - 1
            o_base_v = (v - 1) * nm1 - 1
            o_shift_u = o_base_u - k
            o_shift_v = o_base_v - k

            if has_act:
                for t in range(k + 2, n + 1):
                    append([neg_act, -(o_base_u + t), o_shift_v + t])
                    append([neg_act, -(o_base_v + t), o_shift_u + t])
            else:
                for t in range(k + 2, n + 1):
                    append([-(o_base_u + t), o_shift_v + t])
                    append([-(o_base_v + t), o_shift_u + t])

            if self.use_strengthening:
                x_base_u = num_o + (u - 1) * n
                x_base_v = num_o + (v - 1) * n
                if has_act:
                    for a in range(1, n + 1):
                        if a - k >= 2:
                            append([neg_act, -(x_base_u + a), o_shift_v + a])
                            append([neg_act, -(x_base_v + a), o_shift_u + a])
                        if a + k + 1 <= n:
                            append([neg_act, -(x_base_u + a), -(o_base_v + a + k + 1)])
                            append([neg_act, -(x_base_v + a), -(o_base_u + a + k + 1)])
                else:
                    for a in range(1, n + 1):
                        if a - k >= 2:
                            append([-(x_base_u + a), o_shift_v + a])
                            append([-(x_base_v + a), o_shift_u + a])
                        if a + k + 1 <= n:
                            append([-(x_base_u + a), -(o_base_v + a + k + 1)])
                            append([-(x_base_v + a), -(o_base_u + a + k + 1)])
            maybe_flush()

        # 2-hop distance bounds: d_G(u, v) == 2 => |f(u) - f(v)| <= 2k
        if self.use_2hop and 2 * k < n - 1:
            two_k = 2 * k
            for u, v in self.two_hop_edges:
                o_base_u = (u - 1) * nm1 - 1
                o_base_v = (v - 1) * nm1 - 1
                o_shift_u = o_base_u - two_k
                o_shift_v = o_base_v - two_k
                if has_act:
                    for t in range(two_k + 2, n + 1):
                        append([neg_act, -(o_base_u + t), o_shift_v + t])
                        append([neg_act, -(o_base_v + t), o_shift_u + t])
                else:
                    for t in range(two_k + 2, n + 1):
                        append([-(o_base_u + t), o_shift_v + t])
                        append([-(o_base_v + t), o_shift_u + t])
                maybe_flush()

        if buf:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Encoding time limit exceeded")
            flush_chunk(buf)
            buf.clear()

    def add_base_clauses(self, solver, deadline: Optional[float] = None) -> None:
        self._emit_base_clauses_chunked(solver.append_formula, deadline=deadline)

    def add_bandwidth_clauses(
        self,
        solver,
        k: int,
        act_var: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> None:
        self._emit_bandwidth_clauses_chunked(
            k, solver.append_formula, act_var=act_var, deadline=deadline
        )

    def get_base_clauses(self, deadline: Optional[float] = None) -> List[List[int]]:
        if self._base_clauses is not None:
            return self._base_clauses
        clauses: List[List[int]] = []
        self._emit_base_clauses_chunked(clauses.extend, deadline=deadline)
        self._base_clauses = clauses
        return clauses

    def get_bandwidth_clauses(
        self,
        k: int,
        act_var: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> List[List[int]]:
        clauses: List[List[int]] = []
        self._emit_bandwidth_clauses_chunked(
            k, clauses.extend, act_var=act_var, deadline=deadline
        )
        return clauses

    def encode(self, k: int, deadline: Optional[float] = None) -> List[List[int]]:
        return self.get_base_clauses(deadline=deadline) + self.get_bandwidth_clauses(k, deadline=deadline)

    def decode(self, model: List[int]) -> Dict[int, int]:
        model_set: Set[int] = set(model)
        labeling: Dict[int, int] = {}

        for v in range(1, self.n + 1):
            assigned = None
            for a in range(1, self.n + 1):
                if self.var_X(v, a) in model_set:
                    assigned = a
                    break
            if assigned is not None:
                labeling[v] = assigned
            else:
                curr = 1
                for t in range(2, self.n + 1):
                    if self.var_O(v, t) in model_set:
                        curr = t
                    else:
                        break
                labeling[v] = curr

        return labeling

    def decode_permutation(self, model: List[int]) -> List[int]:
        labeling = self.decode(model)
        return [labeling[v] - 1 for v in range(1, self.n + 1)]

    def compute_bandwidth(self, labeling: Dict[int, int]) -> int:
        if not self.edges:
            return 0
        return max(abs(labeling[u] - labeling[v]) for u, v in self.edges)

    def is_valid_labeling(self, labeling: Dict[int, int]) -> bool:
        return len(labeling) == self.n and set(labeling.values()) == set(range(1, self.n + 1))

    def to_dimacs(self, clauses: List[List[int]]) -> str:
        lines = [f"p cnf {self.total_vars} {len(clauses)}"]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)
