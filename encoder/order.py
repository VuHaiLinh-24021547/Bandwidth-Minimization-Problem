"""
Order Encoding for Bandwidth Minimization Problem (BMP)
-------------------------------------------------------
This module implements the Order Encoding formulation (and Hybrid Encoding)
to encode the 1D Bandwidth Minimization Problem into Propositional Logic CNF.

Theoretical foundation based on:
1. "Exact approaches for Radio mean labeling: ILP and SAT" (VNU-UET, 2026)
2. "Exact SAT Solving for the Two-Dimensional Bandwidth Minimization Problem" (Discrete Optimization, 2026)
3. "Compiling finite linear CSP into SAT" (Tamura et al., Constraints 2009)

Variables:
- Order Variables: O(v, t) <=> f(v) >= t (v in 1..n, t in 2..n)
- Exact Variables: X(v, a) <=> f(v) == a (v in 1..n, a in 1..n)

Constraints:
1. Monotonicity: O(v, t+1) => O(v, t)  (not O(v, t+1) or O(v, t))
2. Channeling: X(v, a) <=> (O(v, a) and not O(v, a+1))
3. AllDifferent / Bijectivity:
   - Exactly one position per vertex (ALO + AMO)
   - Exactly one vertex per position (ALO + AMO)
4. Bandwidth on Edges (u, v) in E:
   - For all t in (k+2..n): (not O(u, t) or O(v, t-k))
   - For all t in (k+2..n): (not O(v, t) or O(u, t-k))
   - Propagation-strengthening resolvents between X and O
"""

from typing import List, Tuple, Dict, Optional, Set
import time


class OrderEncoder:
    """
    Encoder for the 1D Bandwidth Minimization Problem using Order & Hybrid Encoding.
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]], use_strengthening: bool = True):
        """
        Initialize the OrderEncoder.

        :param n: Number of vertices (vertices indexed 1..n).
        :param edges: List of undirected edges (u, v) with 1 <= u, v <= n.
        :param use_strengthening: Whether to add propagation-strengthening resolvents.
        """
        self.n = n
        # Normalize edges to undirected (u < v) and remove duplicates/self-loops
        norm_edges = set()
        for u, v in edges:
            if u != v:
                norm_edges.add((min(u, v), max(u, v)))
        self.edges: List[Tuple[int, int]] = sorted(list(norm_edges))
        self.use_strengthening = use_strengthening

        # Variable mapping:
        # 1. Order variables: O(v, t) for v in 1..n, t in 2..n
        # Count = n * (n - 1)
        # 2. Exact variables: X(v, a) for v in 1..n, a in 1..n
        # Count = n * n
        self.num_order_vars = n * (n - 1)
        self.num_exact_vars = n * n
        self.total_vars = self.num_order_vars + self.num_exact_vars

        # Cached base clauses (independent of bandwidth k)
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
        Construct and return base clauses independent of target bandwidth k.
        Includes:
        - Monotonicity of Order variables
        - Channeling between Exact and Order variables
        - Bijectivity (AllDifferent) across vertices and positions
        """
        if self._base_clauses is not None:
            return self._base_clauses

        clauses: List[List[int]] = []
        n = self.n

        # -------------------------------------------------------------
        # 1. Monotonicity: O(v, t+1) => O(v, t)
        # Clause: ~O(v, t+1) \/ O(v, t)  for t in 2..n-1
        # -------------------------------------------------------------
        for v in range(1, n + 1):
            for t in range(2, n):
                clauses.append([-self.var_O(v, t + 1), self.var_O(v, t)])

        # -------------------------------------------------------------
        # 2. Channeling: X(v, a) <=> O(v, a) /\ ~O(v, a+1)
        # Note: O(v, 1) is constantly TRUE, O(v, n+1) is constantly FALSE.
        # -------------------------------------------------------------
        for v in range(1, n + 1):
            # a = 1: X(v, 1) <=> ~O(v, 2)
            # (~X(v, 1) \/ ~O(v, 2)) /\ (X(v, 1) \/ O(v, 2))
            clauses.append([-self.var_X(v, 1), -self.var_O(v, 2)])
            clauses.append([self.var_X(v, 1), self.var_O(v, 2)])

            # a in 2..n-1:
            for a in range(2, n):
                # X(v, a) => O(v, a)
                clauses.append([-self.var_X(v, a), self.var_O(v, a)])
                # X(v, a) => ~O(v, a+1)
                clauses.append([-self.var_X(v, a), -self.var_O(v, a + 1)])
                # O(v, a) /\ ~O(v, a+1) => X(v, a)
                clauses.append([-self.var_O(v, a), self.var_O(v, a + 1), self.var_X(v, a)])

            # a = n: X(v, n) <=> O(v, n)
            # (~X(v, n) \/ O(v, n)) /\ (X(v, n) \/ ~O(v, n))
            clauses.append([-self.var_X(v, n), self.var_O(v, n)])
            clauses.append([self.var_X(v, n), -self.var_O(v, n)])

        # -------------------------------------------------------------
        # 3. Bijectivity (AllDifferent):
        # (a) Each vertex has at least one label (ALO)
        # (b) Each label is assigned to at least one vertex (ALO)
        # (c) Each label is assigned to at most one vertex (AMO)
        # -------------------------------------------------------------
        # ALO on vertices: \/_a X(v, a)
        for v in range(1, n + 1):
            clauses.append([self.var_X(v, a) for a in range(1, n + 1)])

        # ALO on positions: \/_v X(v, a)
        for a in range(1, n + 1):
            clauses.append([self.var_X(v, a) for v in range(1, n + 1)])

        # AMO on positions: for each position a, at most one vertex
        # Pairwise AMO: ~X(u, a) \/ ~X(v, a) for all u < v
        for a in range(1, n + 1):
            for u in range(1, n + 1):
                for v in range(u + 1, n + 1):
                    clauses.append([-self.var_X(u, a), -self.var_X(v, a)])

        self._base_clauses = clauses
        return clauses

    def get_bandwidth_clauses(self, k: int) -> List[List[int]]:
        """
        Construct clauses enforcing max |f(u) - f(v)| <= k for all (u, v) in E.

        :param k: Bandwidth threshold (0 <= k <= n - 1).
        :return: List of CNF clauses specific to threshold k.
        """
        clauses: List[List[int]] = []
        n = self.n

        # If k >= n - 1, the constraint |f(u) - f(v)| <= k is vacuously satisfied.
        if k >= n - 1:
            return clauses

        for u, v in self.edges:
            # ---------------------------------------------------------
            # Directional Order Implications on Edges:
            # f(u) - f(v) <= k <=> (f(u) >= t => f(v) >= t - k)
            # f(v) - f(u) <= k <=> (f(v) >= t => f(u) >= t - k)
            # For t in k+2..n, (t - k) in 2..n-k:
            # ~O(u, t) \/ O(v, t - k)
            # ~O(v, t) \/ O(u, t - k)
            # ---------------------------------------------------------
            for t in range(k + 2, n + 1):
                clauses.append([-self.var_O(u, t), self.var_O(v, t - k)])
                clauses.append([-self.var_O(v, t), self.var_O(u, t - k)])

            # ---------------------------------------------------------
            # Propagation-strengthening resolvents:
            # If f(u) == a, then a - k <= f(v) <= a + k
            # - If a - k >= 2: ~X(u, a) \/ O(v, a - k)
            # - If a + k + 1 <= n: ~X(u, a) \/ ~O(v, a + k + 1)
            # ---------------------------------------------------------
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
        Generate the full CNF formula for bandwidth <= k.

        :param k: Bandwidth threshold.
        :return: List of all CNF clauses.
        """
        base = self.get_base_clauses()
        bw = self.get_bandwidth_clauses(k)
        return base + bw

    def decode(self, model: List[int]) -> Dict[int, int]:
        """
        Decode a satisfying assignment (model) from SAT solver into a vertex labeling.

        :param model: Satisfying assignment as list of non-zero integer literals.
        :return: Dictionary {vertex: label} where 1 <= vertex, label <= n.
        """
        model_set: Set[int] = set(model)
        labeling: Dict[int, int] = {}

        # Decode via Exact variables X(v, a)
        for v in range(1, self.n + 1):
            assigned = None
            for a in range(1, self.n + 1):
                if self.var_X(v, a) in model_set:
                    assigned = a
                    break
            if assigned is not None:
                labeling[v] = assigned
            else:
                # Fallback: decode via Order variables O(v, t)
                # f(v) = max { t : O(v, t) is True }, default 1
                curr = 1
                for t in range(2, self.n + 1):
                    if self.var_O(v, t) in model_set:
                        curr = t
                    else:
                        break
                labeling[v] = curr

        return labeling

    def compute_bandwidth(self, labeling: Dict[int, int]) -> int:
        """
        Compute the actual maximum bandwidth of a given labeling.
        """
        if not self.edges:
            return 0
        return max(abs(labeling[u] - labeling[v]) for u, v in self.edges)

    def is_valid_labeling(self, labeling: Dict[int, int]) -> bool:
        """
        Verify whether the labeling is a valid permutation of 1..n.
        """
        if len(labeling) != self.n:
            return False
        labels = set(labeling.values())
        return labels == set(range(1, self.n + 1))

    def to_dimacs(self, clauses: List[List[int]]) -> str:
        """
        Format the clauses into standard DIMACS CNF string.
        """
        lines = [f"p cnf {self.total_vars} {len(clauses)}"]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)
