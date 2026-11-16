"""
solver/ILP/CplexMIP.py
======================
CPLEX Mixed-Integer Programming solver for the Bandwidth Minimization Problem
using the local agent.

Uses ``docplex.mp.model.Model`` with binary assignment variables:

* **Variables**:
    - ``x[i, p]`` ∈ {0, 1} — vertex *i* is assigned to position *p*.
    - ``B`` ∈ [0, n−1]     — bandwidth (continuous or integer).
* **Derived position**: ``pos[i] = Σ_p  p · x[i, p]``
* **Constraints**:
    - Each vertex to exactly one position: ``Σ_p x[i,p] = 1   ∀ i``
    - Each position to exactly one vertex: ``Σ_i x[i,p] = 1   ∀ p``
    - ``B ≥ pos[u] − pos[v]``  and  ``B ≥ pos[v] − pos[u]``  ∀ (u,v) ∈ E
    - Symmetry breaking: ``pos[0] ≤ pos[n−1]``
* **Objective**: minimise ``B``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
) -> Dict:
    """Solve BMP with CPLEX MIP using the local agent.

    Parameters
    ----------
    n : int
        Number of vertices.
    edges : list of (int, int)
        Undirected edges (0-based, u < v).
    time_limit : int
        Wall-clock time limit in seconds.
    lower_bound : int, optional
        Theoretical lower bound (if known) to tighten domain and enable early termination.

    Returns
    -------
    dict
        Standardised result dictionary.
    """
    from docplex.mp.model import Model

    # ---- Encoding phase ----
    enc_start = time.perf_counter()

    mdl = Model(name="BMP_CPLEX_MIP")

    # Binary assignment variables: x[i, p] = 1 iff vertex i → position p
    x = {
        (i, p): mdl.binary_var(name=f"x_{i}_{p}")
        for i in range(n)
        for p in range(n)
    }

    # Bandwidth variable
    min_bw = int(lower_bound) if lower_bound is not None and lower_bound > 0 else 0
    B = mdl.integer_var(lb=min_bw, ub=n - 1, name="B")

    # Position expressions: pos[i] = Σ_p  p * x[i, p]
    pos = {
        i: mdl.sum(p * x[i, p] for p in range(n))
        for i in range(n)
    }

    # Assignment constraints
    for i in range(n):
        mdl.add_constraint(
            mdl.sum(x[i, p] for p in range(n)) == 1,
            ctname=f"vertex_{i}_assigned",
        )
    for p in range(n):
        mdl.add_constraint(
            mdl.sum(x[i, p] for i in range(n)) == 1,
            ctname=f"position_{p}_filled",
        )

    # Distance constraints
    for u, v in edges:
        mdl.add_constraint(B >= pos[u] - pos[v], ctname=f"dist_pos_{u}_{v}")
        mdl.add_constraint(B >= pos[v] - pos[u], ctname=f"dist_neg_{u}_{v}")

    # Symmetry breaking
    if n >= 2:
        mdl.add_constraint(pos[0] <= pos[n - 1], ctname="sym_break")

    # Objective
    mdl.minimize(B)

    enc_time = time.perf_counter() - enc_start

    result: Dict = {
        "solver": "CPLEX MIP",
        "status": None,
        "bandwidth": None,
        "permutation": None,
        "encoding_time": enc_time,
        "solve_time": 0.0,
        "gap": None,
    }

    if enc_time > time_limit:
        result["status"] = "TimeLimitReached (encoding)"
        return result

    # ---- Solving phase via local agent ----
    solve_limit = max(1, int(time_limit - enc_time))
    mdl.set_time_limit(solve_limit)
    mdl.context.cplex_parameters.threads = 1
    mdl.context.solver.log_output = False

    solve_start = time.perf_counter()
    solution = mdl.solve(agent="local")
    solve_time = time.perf_counter() - solve_start

    result["solve_time"] = solve_time

    # ---- Extract results ----
    if solution:
        result["status"] = solution.solve_status.name

        perm = [0] * n
        for i in range(n):
            for p in range(n):
                if solution.get_value(x[i, p]) > 0.5:
                    perm[i] = p
                    break

        result["bandwidth"] = int(round(solution.get_value(B)))
        result["permutation"] = perm

        try:
            gap = mdl.solve_details.gap
            result["gap"] = gap * 100.0 if gap is not None else None
        except Exception:
            result["gap"] = None
    else:
        result["status"] = "NoSolution"

    return result
