"""Order and hybrid encoding for the Bandwidth Minimization Problem."""

from typing import List, Tuple, Dict, Optional, Set
import time


class OrderEncoder:
    """Encoder for the 1D Bandwidth Minimization Problem using Order & Hybrid Encoding."""

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
        """
        Initialize OrderEncoder.

        :param n: Number of vertices.
        :param edges: List of undirected edges (0-indexed or 1-indexed).
        :param use_strengthening: Whether to add resolvents between exact and order variables.
        :param use_symmetry_breaking: Whether to add reflection symmetry breaking.
        :param use_degree_bounds: Whether to prune boundaries for high-degree vertices.
        :param use_2hop: Whether to add 2-hop distance bound constraints.
        :param amo_encoding: AMO constraint encoding ('seqcounter' or 'pairwise').
        """
        self.n = n
        self.use_strengthening = use_strengthening
        self.use_symmetry_breaking = use_symmetry_breaking
        self.use_degree_bounds = use_degree_bounds
        self.use_2hop = use_2hop
        self.amo_encoding = amo_encoding.lower()

        # Normalize 0-indexed edges to 1-indexed (1..n) internally
        is_zero_indexed = any(u == 0 or v == 0 for u, v in edges) if edges else False
        if not is_zero_indexed and edges:
            max_v = max(max(u, v) for u, v in edges)
            if max_v < n:
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

        self.edges: List[Tuple[int, int]] = sorted(list(norm_edges))

        # Adjacency and 2-hop neighbor pairs (d_G(u, v) == 2)
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
        self.two_hop_edges: List[Tuple[int, int]] = sorted(list(two_hop_set))

        # Select anchor vertex with maximum degree for symmetry breaking
        self.anchor_vertex = max(range(1, n + 1), key=lambda v: (self.degrees[v], -v))

        # Variable counts and offsets
        self.num_order_vars = n * (n - 1)
        self.num_exact_vars = n * n
        if self.amo_encoding == "seqcounter":
            self.num_aux_vars = n * (n - 1)
        else:
            self.num_aux_vars = 0

        self.total_vars = self.num_order_vars + self.num_exact_vars + self.num_aux_vars
        self._base_clauses: Optional[List[List[int]]] = None

    def var_O(self, v: int, t: int) -> int:
        """DIMACS ID for order variable O(v, t) <=> f(v) >= t (v in 1..n, t in 2..n)."""
        if not (1 <= v <= self.n and 2 <= t <= self.n):
            raise ValueError(f"Invalid indices for O(v={v}, t={t}) with n={self.n}")
        return (v - 1) * (self.n - 1) + (t - 1)

    def var_X(self, v: int, a: int) -> int:
        """DIMACS ID for exact variable X(v, a) <=> f(v) == a (v in 1..n, a in 1..n)."""
        if not (1 <= v <= self.n and 1 <= a <= self.n):
            raise ValueError(f"Invalid indices for X(v={v}, a={a}) with n={self.n}")
        return self.num_order_vars + (v - 1) * self.n + a

    def var_S(self, i: int, a: int) -> int:
        """DIMACS ID for sequential counter aux variable S(i, a) (i in 1..n-1, a in 1..n)."""
        if not (1 <= i <= self.n - 1 and 1 <= a <= self.n):
            raise ValueError(f"Invalid indices for S(i={i}, a={a}) with n={self.n}")
        return self.num_order_vars + self.num_exact_vars + (a - 1) * (self.n - 1) + i

    def get_base_clauses(self) -> List[List[int]]:
        """Construct base CNF clauses independent of target bandwidth k."""
        if self._base_clauses is not None:
            return self._base_clauses

        clauses: List[List[int]] = []
        n = self.n

        # 1. Monotonicity: O(v, t+1) => O(v, t)
        for v in range(1, n + 1):
            for t in range(2, n):
                clauses.append([-self.var_O(v, t + 1), self.var_O(v, t)])

        # 2. Channeling: X(v, a) <=> O(v, a) and not O(v, a+1)
        for v in range(1, n + 1):
            clauses.append([-self.var_X(v, 1), -self.var_O(v, 2)])
            clauses.append([self.var_X(v, 1), self.var_O(v, 2)])

            for a in range(2, n):
                clauses.append([-self.var_X(v, a), self.var_O(v, a)])
                clauses.append([-self.var_X(v, a), -self.var_O(v, a + 1)])
                clauses.append([-self.var_O(v, a), self.var_O(v, a + 1), self.var_X(v, a)])

            clauses.append([-self.var_X(v, n), self.var_O(v, n)])
            clauses.append([self.var_X(v, n), -self.var_O(v, n)])

        # 3. Bijectivity: ALO on vertices and positions
        for v in range(1, n + 1):
            clauses.append([self.var_X(v, a) for a in range(1, n + 1)])

        for a in range(1, n + 1):
            clauses.append([self.var_X(v, a) for v in range(1, n + 1)])

        # AMO on positions
        if self.amo_encoding == "seqcounter":
            for a in range(1, n + 1):
                clauses.append([-self.var_X(1, a), self.var_S(1, a)])
                for i in range(2, n):
                    clauses.append([-self.var_X(i, a), self.var_S(i, a)])
                    clauses.append([-self.var_S(i - 1, a), self.var_S(i, a)])
                    clauses.append([-self.var_S(i - 1, a), -self.var_X(i, a)])
                clauses.append([-self.var_S(n - 1, a), -self.var_X(n, a)])
        else:
            for a in range(1, n + 1):
                for u in range(1, n + 1):
                    for v in range(u + 1, n + 1):
                        clauses.append([-self.var_X(u, a), -self.var_X(v, a)])

        # 4. Symmetry breaking
        if self.use_symmetry_breaking and n >= 2:
            mid_cutoff = (n + 1) // 2 + 1
            if mid_cutoff <= n:
                clauses.append([-self.var_O(self.anchor_vertex, mid_cutoff)])

        self._base_clauses = clauses
        return clauses

    def get_bandwidth_clauses(
        self,
        k: int,
        act_var: Optional[int] = None,
    ) -> List[List[int]]:
        """Construct clauses enforcing |f(u) - f(v)| <= k for all edges (u, v)."""
        clauses: List[List[int]] = []
        n = self.n

        if k >= n - 1:
            return clauses

        # Degree-boundary domain pruning
        if self.use_degree_bounds:
            for u in range(1, n + 1):
                d = self.degrees[u]
                if d > k:
                    min_pos = d - k + 1
                    if 2 <= min_pos <= n:
                        clauses.append([self.var_O(u, min_pos)])
                    max_pos = n - d + k
                    if 1 <= max_pos <= n - 1:
                        clauses.append([-self.var_O(u, max_pos + 1)])

        # Order implications on edges
        for u, v in self.edges:
            for t in range(k + 2, n + 1):
                clauses.append([-self.var_O(u, t), self.var_O(v, t - k)])
                clauses.append([-self.var_O(v, t), self.var_O(u, t - k)])

            # Resolvents between exact and order variables
            if self.use_strengthening:
                for a in range(1, n + 1):
                    if a - k >= 2:
                        clauses.append([-self.var_X(u, a), self.var_O(v, a - k)])
                        clauses.append([-self.var_X(v, a), self.var_O(u, a - k)])
                    if a + k + 1 <= n:
                        clauses.append([-self.var_X(u, a), -self.var_O(v, a + k + 1)])
                        clauses.append([-self.var_X(v, a), -self.var_O(u, a + k + 1)])

        # 2-hop distance bounds for pairs with d_G(u, v) == 2
        if self.use_2hop and 2 * k < n - 1:
            two_k = 2 * k
            for u, v in self.two_hop_edges:
                for t in range(two_k + 2, n + 1):
                    clauses.append([-self.var_O(u, t), self.var_O(v, t - two_k)])
                    clauses.append([-self.var_O(v, t), self.var_O(u, t - two_k)])

        if act_var is not None:
            neg_act = -act_var
            return [[neg_act, *cl] for cl in clauses]

        return clauses

    def encode(self, k: int) -> List[List[int]]:
        """Generate full CNF formula for bandwidth <= k."""
        return self.get_base_clauses() + self.get_bandwidth_clauses(k)

    def decode(self, model: List[int]) -> Dict[int, int]:
        """Decode satisfying assignment into a dictionary {vertex_1based: label_1based}."""
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
        """Decode satisfying assignment into a 0-based permutation array of size n."""
        labeling = self.decode(model)
        perm = [0] * self.n
        for v in range(1, self.n + 1):
            perm[v - 1] = labeling[v] - 1
        return perm

    def compute_bandwidth(self, labeling: Dict[int, int]) -> int:
        """Compute maximum bandwidth of a 1-based labeling."""
        if not self.edges:
            return 0
        return max(abs(labeling[u] - labeling[v]) for u, v in self.edges)

    def is_valid_labeling(self, labeling: Dict[int, int]) -> bool:
        """Verify whether the 1-based labeling is a valid permutation of 1..n."""
        if len(labeling) != self.n:
            return False
        labels = set(labeling.values())
        return labels == set(range(1, self.n + 1))

    def to_dimacs(self, clauses: List[List[int]]) -> str:
        """Format clauses into standard DIMACS CNF string."""
        lines = [f"p cnf {self.total_vars} {len(clauses)}"]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)
