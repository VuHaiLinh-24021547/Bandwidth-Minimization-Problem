"""
Utility functions for Bandwidth Minimization Problem (BMP)
- Parsing Matrix Market instance files (.mtx.rnd)
- Loading lower bounds (lb.csv)
- Logging and writing result CSV files conforming to requirements.md
"""

import os
import csv
from typing import Tuple, List, Dict, Optional


def load_instance(file_path: str) -> Tuple[str, int, int, List[Tuple[int, int]]]:
    """
    Parse a Matrix Market (.mtx.rnd) graph instance file.

    :param file_path: Path to the .mtx.rnd file.
    :return: (instance_name, n, m, edges)
    """
    base_name = os.path.basename(file_path)
    # Strip extension (e.g., arc130.mtx.rnd -> arc130)
    instance_name = base_name.replace(".mtx.rnd", "").replace(".mtx", "")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dim_line = None
    edge_start = 0
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            dim_line = parts
            edge_start = idx + 1
            break

    if not dim_line:
        raise ValueError(f"Could not find dimension line (n n m) in {file_path}")

    n = int(dim_line[0])
    m = int(dim_line[2])

    edges = []
    for line in lines[edge_start:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            u = int(parts[0])
            v = int(parts[1])
            edges.append((u, v))

    return instance_name, n, m, edges


def load_lower_bounds(lb_csv_path: str = "input/lb.csv") -> Dict[str, int]:
    """
    Load lower bounds from lb.csv into a dictionary: {instance_name: lb}.
    """
    lb_dict = {}
    if not os.path.exists(lb_csv_path):
        return lb_dict

    with open(lb_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and len(row) >= 2:
                inst = row[0].strip()
                try:
                    val = int(row[1].strip())
                    lb_dict[inst] = val
                except ValueError:
                    continue
    return lb_dict


def write_result_csv(
    output_path: str,
    results: List[Dict],
    include_extra_metrics: bool = True
):
    """
    Write experiment results to CSV according to requirements.md:
    Columns required: instance, n, edges, lb, encoding_time, solve_stime
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fieldnames = ["instance", "n", "edges", "lb", "encoding_time", "solve_stime"]
    if include_extra_metrics:
        fieldnames.extend(["bandwidth", "status", "solver"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
