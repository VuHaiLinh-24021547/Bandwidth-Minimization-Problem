"""Incremental SAT dispatcher routing to Order or Pairwise modules."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.SAT.orderIncrementalSAT import get_rcm_upper_bound, solve as solve_order
from solver.SAT.pairwiseIncrementalSAT import solve as solve_pairwise


def solve(
    n: int,
    edges: List[Tuple[int, int]],
    time_limit: int = 60,
    lower_bound: Optional[int] = None,
    solver_name: str = "glucose4",
    encoding: str = "order",
    **kwargs,
) -> Dict:
    enc = encoding.strip().lower()
    if enc == "order":
        return solve_order(
            n,
            edges,
            time_limit=time_limit,
            lower_bound=lower_bound,
            solver_name=solver_name,
            **kwargs,
        )
    if enc == "pairwise":
        return solve_pairwise(
            n,
            edges,
            time_limit=time_limit,
            lower_bound=lower_bound,
            solver_name=solver_name,
            **kwargs,
        )
    raise ValueError(f"Unsupported SAT encoding '{encoding}'. Expected 'order' or 'pairwise'.")


if __name__ == "__main__":
    from utils.hb_parser import load_hb_file
    from utils.lb_loader import get_instance_stem, load_lb_mapping, lookup_lb

    test_file = os.path.join(PROJECT_ROOT, "input/small_instances/bcspwr01.mtx.rnd")
    n, edges = load_hb_file(test_file)
    stem = get_instance_stem(test_file)
    lb = lookup_lb(stem, load_lb_mapping())
    print(solve(n, edges, time_limit=60, lower_bound=lb, encoding="order"))
