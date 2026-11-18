"""
Multiple (Non-Incremental) SAT Solver for Bandwidth Minimization Problem (BMP)
-----------------------------------------------------------------------------
Uses Order Encoding to solve BMP via Binary Search over bandwidth bounds [LB, UB].
For each candidate bandwidth k, a new SAT formula and solver instance are created.

Requirements adherence:
- Uses time.perf_counter() to measure encoding_time and solve_stime.
- Supports timeout interruption if encoding or solving exceeds limit (via threading.Timer & solver.interrupt).
- Formats output according to requirements.md: instance, n, edges, lb, encoding_time, solve_stime.
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


class MultipleSATSolver:
    """
    Non-incremental SAT solver executing Binary Search over target bandwidth k.
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
        Execute Binary Search to find the minimum bandwidth.
        Tracks encoding_time and solve_stime with timeout interruption guard.
        """
        total_encoding_time = 0.0
        total_solve_time = 0.0
        start_wall_time = time.perf_counter()

        low = self.lb
        high = self.ub
        best_bw = high
        best_labeling = None
        status = "UNKNOWN"

        encoder = OrderEncoder(self.n, self.edges)

        # Quick check if low >= high
        if low >= high:
            clauses, enc_t = self._timed_encode(encoder, low)
            total_encoding_time += enc_t
            rem = max(0.01, self.timeout - (time.perf_counter() - start_wall_time))
            sat, model, s_t = self._timed_solve(clauses, rem)
            total_solve_time += s_t
            if sat is True:
                best_bw = low
                best_labeling = encoder.decode(model)
                status = "OPTIMAL"
            elif sat is False:
                best_bw = high
                status = "INFEASIBLE_LB"
            else:
                status = "TIMEOUT"
            return {
                "opt_bandwidth": best_bw,
                "encoding_time": total_encoding_time,
                "solve_stime": total_solve_time,
                "status": status,
                "labeling": best_labeling
            }

        while low <= high:
            # Check global timeout
            elapsed = time.perf_counter() - start_wall_time
            if elapsed >= self.timeout:
                status = "TIMEOUT" if best_labeling is None else "FEASIBLE"
                break

            mid = (low + high) // 2

            # 1. Encoding with timeout guard
            t_enc0 = time.perf_counter()
            clauses = encoder.encode(mid)
            enc_t = time.perf_counter() - t_enc0
            total_encoding_time += enc_t

            elapsed = time.perf_counter() - start_wall_time
            if elapsed >= self.timeout:
                status = "TIMEOUT" if best_labeling is None else "FEASIBLE"
                break

            # 2. Solving with timeout guard
            remaining_time = max(0.01, self.timeout - elapsed)
            sat, model, s_t = self._timed_solve(clauses, remaining_time)
            total_solve_time += s_t

            if sat is None or (time.perf_counter() - start_wall_time) >= self.timeout:
                # Interrupted due to timeout
                status = "TIMEOUT" if best_labeling is None else "FEASIBLE"
                break

            if sat is True:
                labeling = encoder.decode(model)
                actual_bw = encoder.compute_bandwidth(labeling)
                if actual_bw <= best_bw:
                    best_bw = actual_bw
                    best_labeling = labeling
                # Tighten upper bound
                high = min(mid - 1, actual_bw - 1)
                status = "OPTIMAL"
            else:
                # mid is infeasible
                low = mid + 1

        if low > high and status != "TIMEOUT" and status != "FEASIBLE":
            status = "OPTIMAL"

        return {
            "opt_bandwidth": best_bw,
            "encoding_time": total_encoding_time,
            "solve_stime": total_solve_time,
            "status": status,
            "labeling": best_labeling
        }

    def _timed_encode(self, encoder: OrderEncoder, k: int):
        t0 = time.perf_counter()
        clauses = encoder.encode(k)
        t_enc = time.perf_counter() - t0
        return clauses, t_enc

    def _timed_solve(self, clauses: list, timeout_sec: float):
        t0 = time.perf_counter()
        solver = Solver(name=self.solver_name, bootstrap_with=clauses)
        timer = threading.Timer(timeout_sec, solver.interrupt)
        timer.start()
        try:
            sat = solver.solve_limited(expect_interrupt=True)
            model = solver.get_model() if sat is True else None
        finally:
            timer.cancel()
            s_t = time.perf_counter() - t0
            solver.delete()
        return sat, model, s_t


def solve_file(file_path: str, lb_map: dict, timeout: float = 300.0, solver_name: str = "glucose4") -> Dict[str, Any]:
    inst_name, n, m, edges = load_instance(file_path)
    lb = lb_map.get(inst_name, 1)

    solver = MultipleSATSolver(n=n, edges=edges, lb=lb, timeout=timeout, solver_name=solver_name)
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
        "solver": "MultipleSAT_Order"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MultipleSAT with Order Encoding.")
    parser.add_argument("--input", type=str, default="input/small_instances/bcspwr01.mtx.rnd")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    lb_map = load_lower_bounds()
    result = solve_file(args.input, lb_map, timeout=args.timeout)
    print("Result:", result)
