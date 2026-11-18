from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 120,
    lower_bound: Optional[int] = None,
) -> Dict:
    from docplex.cp.model import CpoModel

    enc_start = time.perf_counter()

    mdl = CpoModel(name="BMP_CPLEX_CP")

    # Decision variables
    min_bw = int(lower_bound) if lower_bound is not None and lower_bound > 0 else 0
    pos = [mdl.integer_var(0, n - 1, f"pos_{i}") for i in range(n)]
    bw = mdl.integer_var(min_bw, n - 1, "bw")

    #  each vertex gets a unique position
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

    solve_limit = max(1, int(time_limit - enc_time))
    solve_kwargs = {
        "agent": "local",
        "TimeLimit": solve_limit,
        "LogVerbosity": "Quiet",
    }

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
