"""
solver/ILP/gurobi.py
====================
Gurobi Mixed-Integer Programming solver for the Bandwidth Minimization Problem.
Configured to use the academic license in gurobi.lic.

Implements the same binary-assignment MIP formulation as ``CplexMIP.py``:

* **Variables**:
    - ``x[i, p]`` ∈ {0, 1} — vertex *i* is assigned to position *p*.
    - ``B`` ∈ [0, n−1]     — bandwidth (continuous).
* **Derived position**: ``pos[i] = Σ_p  p · x[i, p]``
* **Constraints**:
    - ``Σ_p x[i,p] = 1   ∀ i``  (each vertex → one position)
    - ``Σ_i x[i,p] = 1   ∀ p``  (each position → one vertex)
    - ``B ≥ pos[u] − pos[v]``  and  ``B ≥ pos[v] − pos[u]``  ∀ (u,v) ∈ E
    - Symmetry breaking: ``pos[0] ≤ pos[n−1]``
* **Objective**: minimise ``B``.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
) -> Dict:
    """Solve BMP with Gurobi MIP using academic license.

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
    import gurobipy as gp
    from gurobipy import GRB, quicksum

    # ---- Encoding phase ----
    enc_start = time.perf_counter()

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)  # suppress output
        env.setParam("Threads", 1)
        env.start()

        model = gp.Model("BMP_Gurobi", env=env)

        # Binary assignment variables
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

        # Bandwidth variable
        min_bw = float(lower_bound) if lower_bound is not None and lower_bound > 0 else 0.0
        B = model.addVar(lb=min_bw, ub=n - 1, vtype=GRB.CONTINUOUS, name="B")

        # Position expressions: pos[i] = Σ_p  p * x[i, p]
        pos = {
            i: quicksum(p * x[i, p] for p in range(n))
            for i in range(n)
        }

        # Assignment constraints
        for i in range(n):
            model.addConstr(
                quicksum(x[i, p] for p in range(n)) == 1,
                name=f"vertex_{i}",
            )
        for p in range(n):
            model.addConstr(
                quicksum(x[i, p] for i in range(n)) == 1,
                name=f"position_{p}",
            )

        # Distance constraints
        for u, v in edges:
            model.addConstr(B >= pos[u] - pos[v], name=f"dist_pos_{u}_{v}")
            model.addConstr(B >= pos[v] - pos[u], name=f"dist_neg_{u}_{v}")

        # Symmetry breaking
        if n >= 2:
            model.addConstr(pos[0] <= pos[n - 1], name="sym_break")

        # Objective
        model.setObjective(B, GRB.MINIMIZE)
        model.update()

        enc_time = time.perf_counter() - enc_start

        result: Dict = {
            "solver": "Gurobi MIP",
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
        model.Params.TimeLimit = solve_limit
        if lower_bound is not None and lower_bound > 0:
            model.Params.BestObjStop = float(lower_bound)

        solve_start = time.perf_counter()
        model.optimize()
        solve_time = time.perf_counter() - solve_start

        result["solve_time"] = solve_time

        # ---- Extract results ----
        status_code = model.Status
        status_map = {
            GRB.OPTIMAL: "Optimal",
            GRB.INFEASIBLE: "Infeasible",
            GRB.INF_OR_UNBD: "InfOrUnbounded",
            GRB.UNBOUNDED: "Unbounded",
            GRB.TIME_LIMIT: "TimeLimitReached",
            GRB.NODE_LIMIT: "NodeLimit",
            GRB.SOLUTION_LIMIT: "SolutionLimit",
            GRB.SUBOPTIMAL: "SubOptimal",
            GRB.USER_OBJ_LIMIT: "Optimal",
        }
        result["status"] = status_map.get(status_code, f"GurobiStatus_{status_code}")

        if model.SolCount > 0:
            perm = [0] * n
            for i in range(n):
                for p in range(n):
                    if x[i, p].X > 0.5:
                        perm[i] = p
                        break

            result["bandwidth"] = int(round(B.X))
            result["permutation"] = perm

            try:
                result["gap"] = model.MIPGap * 100.0
            except Exception:
                result["gap"] = None
        else:
            if result["status"] is None:
                result["status"] = "NoSolution"

        return result
