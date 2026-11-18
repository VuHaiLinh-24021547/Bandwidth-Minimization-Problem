"""
Order Encoding for Bandwidth Minimization Problem (BMP)
-------------------------------------------------------
This module implements the Order Encoding formulation (and Hybrid Encoding)
to encode the 1D Bandwidth Minimization Problem into Propositional Logic CNF.

Theoretical foundation based on:
1. "Exact approaches for Radio mean labeling: ILP and SAT" (VNU-UET, 2026)
2. "Exact SAT Solving for the Two-Dimensional Bandwidth Minimization Problem" (Discrete Optimization, 2026)
3. "Compiling finite linear CSP into SAT" (Tamura et al., Constraints 2009)

Enhancements:
- Seamless support for 0-indexed and 1-indexed graph representations.
- Global Reflection Symmetry Breaking: Restricts anchor vertex v_anchor to the first half of labels.
- Degree-Boundary Domain Pruning: Tightens domains for high-degree vertices based on adjacency capacity.
- Propagation-strengthening resolvents between Exact and Order variables.
"""

from typing import List, Tuple, Dict, Optional, Set
import time


class OrderEncoder:
    """
    Encoder for the 1D Bandwidth Minimization Problem using Order & Hybrid Encoding.
    """

    def __init__(
        self,
        n: int,
        edges: List[Tuple[int, int]],
        use_strengthening: bool = True,
        use_symmetry_breaking: bool = True,
        use_degree_bounds: bool = True,
    ):
        """
        Initialize the OrderEncoder.

        :param n: Number of vertices.
        :param edges: List of undirected edges. Supports both 0-indexed and 1-indexed edges.
        :param use_strengthening: Whether to add propagation-strengthening resolvents.
        :param use_symmetry_breaking: Whether to add global reflection symmetry breaking.
        :param use_degree_bounds: Whether to prune boundaries for high-degree vertices.
        """
        self.n = n
        self.use_strengthening = use_strengthening
        self.use_symmetry_breaking = use_symmetry_breaking
        self.use_degree_bounds = use_degree_bounds

        # Detect 0-indexed vs 1-indexed:
        # If any endpoint is 0 or all endpoints <= n-1, normalize to 1-indexed (1..n) internally
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

        # Select anchor vertex for symmetry breaking:
        # Vertex with maximum degree; tie-break by index
        self.anchor_vertex = max(range(1, n + 1), key=lambda v: (self.degrees[v], -v))

        # Variable mapping:
        # 1. Order variables: O(v, t) for v in 1..n, t in 2..n -> Count = n * (n - 1)
        # 2. Exact variables: X(v, a) for v in 1..n, a in 1..n -> Count = n * n
        self.num_order_vars = n * (n - 1)
        self.num_exact_vars = n * n
        self.total_vars = self.num_order_vars + self.num_exact_vars

        self._base_clauses: Optional[List[List[int]]] = None

    def var_O(self, v: int, t: int) -> int:
        """
        DIMACS ID for Order variable O(v, t) <=> f(v) >= t.
        Valid for v in 1..n, t in 2..n.
        """
        if not (1 <= v <= self.n and 2 <= t <= self.n):
            raise ValueError(f"Invalid indices for O(v={v}, t={t}) with n={self.n}")
        return (v - 1) * (self.n - 1) + (t - 1)

    def var_X(self, v: int, a: int) -> int:
        """
        DIMACS ID for Exact variable X(v, a) <=> f(v) == a.
        Valid for v in 1..n, a in 1..n.
        """
        if not (1 <= v <= self.n and 1 <= a <= self.n):
            raise ValueError(f"Invalid indices for X(v={v}, a={a}) with n={self.n}")
        return self.num_order_vars + (v - 1) * self.n + a

    def get_base_clauses(self) -> List[List[int]]:
        """
        Construct base clauses independent of target bandwidth k:
        - Monotonicity: O(v, t+1) => O(v, t)
        - Channeling: X(v, a) <=> O(v, a) /\ ~O(v, a+1)
        - Bijectivity (AllDifferent): ALO on vertices and positions, AMO on positions
        - Global Reflection Symmetry Breaking (optional)
        """
        if self._base_clauses is not None:
            return self._base_clauses

        clauses: List[List[int]] = []
        n = self.n

        # 1. Monotonicity
        for v in range(1, n + 1):
            for t in range(2, n):
                clauses.append([-self.var_O(v, t + 1), self.var_O(v, t)])

        # 2. Channeling
        for v in range(1, n + 1):
            # a = 1: X(v, 1) <=> ~O(v, 2)
            clauses.append([-self.var_X(v, 1), -self.var_O(v, 2)])
            clauses.append([self.var_X(v, 1), self.var_O(v, 2)])

            for a in range(2, n):
                clauses.append([-self.var_X(v, a), self.var_O(v, a)])
                clauses.append([-self.var_X(v, a), -self.var_O(v, a + 1)])
                clauses.append([-self.var_O(v, a), self.var_O(v, a + 1), self.var_X(v, a)])

            # a = n: X(v, n) <=> O(v, n)
            clauses.append([-self.var_X(v, n), self.var_O(v, n)])
            clauses.append([self.var_X(v, n), -self.var_O(v, n)])

        # 3. Bijectivity (AllDifferent)
        # ALO on vertices
        for v in range(1, n + 1):
            clauses.append([self.var_X(v, a) for a in range(1, n + 1)])

        # ALO on positions
        for a in range(1, n + 1):
            clauses.append([self.var_X(v, a) for v in range(1, n + 1)])

        # AMO on positions (pairwise)
        for a in range(1, n + 1):
            for u in range(1, n + 1):
                for v in range(u + 1, n + 1):
                    clauses.append([-self.var_X(u, a), -self.var_X(v, a)])

        # 4. Global Reflection Symmetry Breaking:
        # For anchor vertex a_v: f(a_v) <= ceil(n/2)
        # In Order Encoding: ~O(a_v, ceil(n/2) + 1)
        if self.use_symmetry_breaking and n >= 2:
            mid_cutoff = (n + 1) // 2 + 1
            if mid_cutoff <= n:
                clauses.append([-self.var_O(self.anchor_vertex, mid_cutoff)])

        self._base_clauses = clauses
        return clauses

    def get_bandwidth_clauses(self, k: int) -> List[List[int]]:
        """
        Construct clauses enforcing max |f(u) - f(v)| <= k for all (u, v) in E.
        """
        clauses: List[List[int]] = []
        n = self.n

        if k >= n - 1:
            return clauses

        # 1. Degree-boundary pruning clauses
        # If vertex u has degree d(u) > k:
        # f(u) >= d(u) - k + 1  =>  O(u, d(u) - k + 1)
        # f(u) <= n - d(u) + k  =>  ~O(u, n - d(u) + k + 1)
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

        # 2. Directional Order Implications on Edges
        for u, v in self.edges:
            for t in range(k + 2, n + 1):
                clauses.append([-self.var_O(u, t), self.var_O(v, t - k)])
                clauses.append([-self.var_O(v, t), self.var_O(u, t - k)])

            # 3. Propagation-strengthening resolvents
            if self.use_strengthening:
                for a in range(1, n + 1):
                    if a - k >= 2:
                        clauses.append([-self.var_X(u, a), self.var_O(v, a - k)])
                        clauses.append([-self.var_X(v, a), self.var_O(u, a - k)])
                    if a + k + 1 <= n:
                        clauses.append([-self.var_X(u, a), -self.var_O(v, a + k + 1)])
                        clauses.append([-self.var_X(v, a), -self.var_O(u, a + k + 1)])

        return clauses

    def encode(self, k: int) -> List[List[int]]:
        """
        Generate full CNF formula for bandwidth <= k.
        """
        return self.get_base_clauses() + self.get_bandwidth_clauses(k)

    def decode(self, model: List[int]) -> Dict[int, int]:
        """
        Decode satisfying assignment into a dictionary {vertex_1based: label_1based}.
        """
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
        """
        Decode model into a 0-based permutation array of size n:
        perm[i] is the 0-based position of vertex i (for i in 0..n-1).
        Compatible with the project's CP and ILP solvers.
        """
        labeling = self.decode(model)
        perm = [0] * self.n
        for v in range(1, self.n + 1):
            # v is 1-based, label is 1-based -> store 0-based
            perm[v - 1] = labeling[v] - 1
        return perm

    def compute_bandwidth(self, labeling: Dict[int, int]) -> int:
        """
        Compute maximum bandwidth of a 1-based labeling.
        """
        if not self.edges:
            return 0
        return max(abs(labeling[u] - labeling[v]) for u, v in self.edges)

    def is_valid_labeling(self, labeling: Dict[int, int]) -> bool:
        """
        Verify whether the 1-based labeling is a valid permutation of 1..n.
        """
        if len(labeling) != self.n:
            return False
        labels = set(labeling.values())
        return labels == set(range(1, self.n + 1))

    def to_dimacs(self, clauses: List[List[int]]) -> str:
        """
        Format clauses into standard DIMACS CNF string.
        """
        lines = [f"p cnf {self.total_vars} {len(clauses)}"]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)
