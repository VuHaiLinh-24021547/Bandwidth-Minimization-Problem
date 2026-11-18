"""
Incremental SAT Solver for Bandwidth Minimization Problem (BMP)
--------------------------------------------------------------
Uses Order Encoding to solve BMP via Descending Threshold Search on a
single, persistent SAT solver instance.

Theoretical framework:
- Base constraints (Monotonicity, Channeling, AllDifferent) are asserted once.
- Bandwidth constraints Delta_k are added cumulatively as target bandwidth k decreases.
- Learned conflict clauses are retained across threshold steps, accelerating certification.
- Smart Jump: when a model has bandwidth B <= k, next threshold jumps directly to B - 1.
- Early exit: when B reaches Lower Bound (LB), optimality is immediately certified.

Requirements adherence:
- Uses time.perf_counter() for encoding_time and solve_stime.
- Timeout interruption guard via threading.Timer and solver.interrupt.
- Output formatting conforms to requirements.md: instance, n, edges, lb, encoding_time, solve_stime.
"""

import sys
import os
import time
import threading
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from encoder.order import OrderEncoder
from utils import load_instance, load_lower_bounds, write_result_csv
from pysat.solvers import Glucose4, Solver


class IncrementalSATSolver:
    """
    Incremental SAT solver maintaining persistent clauses and solver state.
    """

    def __init__(
        self,
        n: int,
        edges: list,
        lb: int = 1,
        ub: Optional[int] = None,
        timeout: float = 300.0,
        solver_name: str = "glucose4"
    ):
        self.n = n
        self.edges = edges
        self.lb = max(1, lb)
        self.ub = min(n - 1, ub) if ub is not None else n - 1
        self.timeout = timeout
        self.solver_name = solver_name

    def solve(self) -> Dict[str, Any]:
        """
        Execute incremental descending search.
        """
        total_encoding_time = 0.0
        total_solve_time = 0.0
        start_wall_time = time.perf_counter()

        # Step 1: Initialize encoder and build base clauses
        t_enc_start = time.perf_counter()
        encoder = OrderEncoder(self.n, self.edges)
        base_clauses = encoder.get_base_clauses()
        enc_base_time = time.perf_counter() - t_enc_start
        total_encoding_time += enc_base_time

        # Step 2: Initialize persistent SAT solver
        solver = Solver(name=self.solver_name, bootstrap_with=base_clauses)

        k = self.ub
        best_bw = self.ub
        best_labeling = None
        status = "UNKNOWN"

        try:
            while k >= 1:
                # Check global timeout
                elapsed = time.perf_counter() - start_wall_time
                if elapsed >= self.timeout:
                    status = "TIMEOUT" if best_labeling is None else "FEASIBLE"
                    break

                # Step 3: Add threshold clauses Delta_k
                t_delta = time.perf_counter()
                bw_clauses = encoder.get_bandwidth_clauses(k)
                for cl in bw_clauses:
                    solver.add_clause(cl)
                total_encoding_time += (time.perf_counter() - t_delta)

                elapsed = time.perf_counter() - start_wall_time
                if elapsed >= self.timeout:
                    status = "TIMEOUT" if best_labeling is None else "FEASIBLE"
                    break

                # Step 4: Solve with persistent state and timeout guard
                remaining_time = max(0.01, self.timeout - elapsed)
                timer = threading.Timer(remaining_time, solver.interrupt)
                timer.start()

                t_s = time.perf_counter()
                sat = None
                try:
                    sat = solver.solve_limited(expect_interrupt=True)
                finally:
                    timer.cancel()
                    solver.clear_interrupt()
                    s_t = time.perf_counter() - t_s
                    total_solve_time += s_t

                if sat is None or (time.perf_counter() - start_wall_time) >= self.timeout:
                    # Timeout reached
                    status = "TIMEOUT" if best_labeling is None else "FEASIBLE"
                    break

                if sat is True:
                    model = solver.get_model()
                    labeling = encoder.decode(model)
                    actual_bw = encoder.compute_bandwidth(labeling)

                    if actual_bw <= best_bw:
                        best_bw = actual_bw
                        best_labeling = labeling

                    # Check lower bound certificate
                    if best_bw <= self.lb:
                        status = "OPTIMAL"
                        break

                    # Smart jump: jump directly to best_bw - 1
                    k = best_bw - 1
                else:
                    # k is UNSAT, therefore best_bw is certified OPTIMAL!
                    status = "OPTIMAL"
                    break

            if status != "TIMEOUT" and status != "FEASIBLE":
                status = "OPTIMAL"

        finally:
            solver.delete()

        return {
            "opt_bandwidth": best_bw,
            "encoding_time": total_encoding_time,
            "solve_stime": total_solve_time,
            "status": status,
            "labeling": best_labeling
        }


def solve_file(file_path: str, lb_map: dict, timeout: float = 300.0, solver_name: str = "glucose4") -> Dict[str, Any]:
    inst_name, n, m, edges = load_instance(file_path)
    lb = lb_map.get(inst_name, 1)

    solver = IncrementalSATSolver(n=n, edges=edges, lb=lb, timeout=timeout, solver_name=solver_name)
    res = solver.solve()

    return {
        "instance": inst_name,
        "n": n,
        "edges": m,
        "lb": lb,
        "encoding_time": round(res["encoding_time"], 4),
        "solve_stime": round(res["solve_stime"], 4),
        "bandwidth": res["opt_bandwidth"],
        "status": res["status"],
        "solver": "IncrementalSAT_Order"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run IncrementalSAT with Order Encoding.")
    parser.add_argument("--input", type=str, default="input/small_instances/bcspwr01.mtx.rnd")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    lb_map = load_lower_bounds()
    result = solve_file(args.input, lb_map, timeout=args.timeout)
    print("Result:", result)
