"""
solver/CP/CplexCP.py
====================
CPLEX Constraint Programming solver for the Bandwidth Minimization Problem
using the local agent.

Uses ``docplex.cp.model.CpoModel`` to formulate the BMP as:

* **Variables**: ``pos[i]`` ∈ {0, …, n−1}  (position of vertex i),
  ``bw`` ∈ {0, …, n−1}  (bandwidth).
* **Constraints**:
    - ``AllDifferent(pos)``
    - ``bw ≥ |pos[u] − pos[v]|``  for each edge (u, v)
    - Symmetry breaking: ``pos[0] < pos[n−1]``  (for n ≥ 2)
* **Objective**: minimise ``bw``.
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
    """Solve BMP with CPLEX CP Optimizer using the local agent.

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
    from docplex.cp.model import CpoModel

    # ---- Encoding phase ----
    enc_start = time.perf_counter()

    mdl = CpoModel(name="BMP_CPLEX_CP")

    # Decision variables
    min_bw = int(lower_bound) if lower_bound is not None and lower_bound > 0 else 0
    pos = [mdl.integer_var(0, n - 1, f"pos_{i}") for i in range(n)]
    bw = mdl.integer_var(min_bw, n - 1, "bw")

    # All-different: each vertex gets a unique position
    mdl.add(mdl.all_diff(pos))

    # Bandwidth constraints
    for u, v in edges:
        mdl.add(bw >= mdl.abs(pos[u] - pos[v]))

    # Symmetry breaking
    if n >= 2:
        mdl.add(pos[0] < pos[n - 1])

    # Objective
    mdl.minimize(bw)

    enc_time = time.perf_counter() - enc_start

    result: Dict = {
        "solver": "CPLEX CP",
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
    import os
    import shutil

    solve_limit = max(1, int(time_limit - enc_time))
    solve_kwargs = {
        "agent": "local",
        "TimeLimit": solve_limit,
        "LogVerbosity": "Quiet",
    }
    if not shutil.which("cpoptimizer"):
        candidates = [
            os.path.abspath("ibm/ILOG/CPLEX_Studio222/cpoptimizer/bin/x86-64_linux/cpoptimizer"),
            os.path.abspath(".venv/bin/cpoptimizer"),
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                solve_kwargs["execfile"] = c
                break

    solve_start = time.perf_counter()
    solution = mdl.solve(**solve_kwargs)
    solve_time = time.perf_counter() - solve_start

    result["solve_time"] = solve_time

    # ---- Extract results ----
    if solution:
        status_str = solution.get_solve_status()
        result["status"] = status_str

        perm = [solution.get_value(pos[i]) for i in range(n)]
        bw_val = solution.get_value(bw)

        result["bandwidth"] = int(bw_val)
        result["permutation"] = [int(p) for p in perm]

        try:
            obj_val = solution.get_objective_value()
            obj_bound = solution.get_objective_bound()
            if obj_val and obj_val > 0:
                result["gap"] = abs(obj_val - obj_bound) / obj_val * 100.0
            else:
                result["gap"] = 0.0
        except Exception:
            result["gap"] = None
    else:
        result["status"] = "NoSolution"

    return result
