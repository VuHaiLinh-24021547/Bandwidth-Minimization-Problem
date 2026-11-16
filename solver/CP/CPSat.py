"""
solver/CP/CPSat.py
==================
Google OR-Tools CP-SAT solver for the Bandwidth Minimization Problem.

Uses ``ortools.sat.python.cp_model`` to formulate the BMP as:

* **Variables**: ``pos[i]`` ∈ {0, …, n−1},  ``bw`` ∈ {0, …, n−1}.
* **Constraints**:
    - ``AddAllDifferent(pos)``
    - For each (u, v) ∈ E: auxiliary ``diff`` and ``abs_diff`` variables
      with ``AddAbsEquality(abs_diff, diff)`` and ``bw ≥ abs_diff``.
    - Symmetry breaking: ``pos[0] < pos[n−1]``
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
    """Solve BMP with Google OR-Tools CP-SAT.

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
    """
    from ortools.sat.python import cp_model

    # ---- Encoding phase ----
    enc_start = time.perf_counter()

    model = cp_model.CpModel()

    # Decision variables
    min_bw = int(lower_bound) if lower_bound is not None and lower_bound > 0 else 0
    pos = [model.NewIntVar(0, n - 1, f"pos_{i}") for i in range(n)]
    bw = model.NewIntVar(min_bw, n - 1, "bw")

    # All-different
    model.AddAllDifferent(pos)

    # Bandwidth constraints via auxiliary variables
    for idx, (u, v) in enumerate(edges):
        diff = model.NewIntVar(-(n - 1), n - 1, f"diff_{idx}")
        abs_diff = model.NewIntVar(0, n - 1, f"abs_diff_{idx}")
        model.Add(diff == pos[u] - pos[v])
        model.AddAbsEquality(abs_diff, diff)
        model.Add(bw >= abs_diff)

    # Symmetry breaking
    if n >= 2:
        model.Add(pos[0] < pos[n - 1])

    # Objective
    model.Minimize(bw)

    enc_time = time.perf_counter() - enc_start

    result: Dict = {
        "solver": "OR-Tools CP-SAT",
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

    # ---- Solving phase ----
    solve_limit = max(1, int(time_limit - enc_time))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solve_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_workers = 1

    solve_start = time.perf_counter()
    status = solver.Solve(model)
    solve_time = time.perf_counter() - solve_start

    result["solve_time"] = solve_time

    # ---- Extract results ----
    status_map = {
        cp_model.OPTIMAL: "Optimal",
        cp_model.FEASIBLE: "Feasible",
        cp_model.INFEASIBLE: "Infeasible",
        cp_model.MODEL_INVALID: "ModelInvalid",
        cp_model.UNKNOWN: "Unknown",
    }
    result["status"] = status_map.get(status, "Unknown")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        perm = [solver.Value(pos[i]) for i in range(n)]
        bw_val = solver.Value(bw)

        result["bandwidth"] = int(bw_val)
        result["permutation"] = perm

        best_bound = solver.BestObjectiveBound()
        obj_val = solver.ObjectiveValue()
        if obj_val > 0:
            result["gap"] = abs(obj_val - best_bound) / obj_val * 100.0
        else:
            result["gap"] = 0.0

    return result
