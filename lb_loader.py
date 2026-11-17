"""
Utilities for loading and querying lower bounds from CSV (input/lb.csv)
and common graph solution verification helpers.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Tuple


def get_instance_stem(filepath: str) -> str:
    """Extract instance name without .mtx.rnd (or other extensions)."""
    filename = os.path.basename(filepath)
    if filename.endswith(".mtx.rnd"):
        return filename[:-8]
    elif filename.endswith(".rnd"):
        return filename[:-4]
    elif filename.endswith(".mtx"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def get_instance_category(filepath: str) -> str:
    """Determine whether an instance is small or large based on its directory path."""
    norm = filepath.replace("\\", "/").lower()
    if "large_instances" in norm or "large instances" in norm:
        return "large_instances"
    elif "small_instances" in norm or "small instances" in norm:
        return "small_instances"
    else:
        parent = os.path.basename(os.path.dirname(filepath))
        return parent if parent not in ("", ".", "input") else "small_instances"


def load_lb_mapping(csv_path: str = "input/lb.csv") -> Dict[str, int]:
    """Load lower bounds mapping from a CSV file (instance -> LB)."""
    mapping: Dict[str, int] = {}
    if not os.path.exists(csv_path):
        return mapping
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            name, val = row[0].strip(), row[1].strip()
            if name.lower() == "instance":
                continue
            try:
                mapping[name] = int(val)
            except ValueError:
                continue
    return mapping


def lookup_lb(instance_name: str, lb_table: Dict[str, int]) -> Optional[int]:
    """Look up LB for an instance name, handling underscore differences."""
    if instance_name in lb_table:
        return lb_table[instance_name]
    norm = re.sub(r"_+", "_", instance_name)
    if norm in lb_table:
        return lb_table[norm]
    lower = instance_name.lower()
    norm_lower = norm.lower()
    for k, v in lb_table.items():
        if k.lower() in (lower, norm_lower):
            return v
    return None


def compute_original_bandwidth(n: int, edges: List[Tuple[int, int]]) -> int:
    """Bandwidth under the identity permutation (pi(i) = i)."""
    if not edges:
        return 0
    return max(abs(u - v) for u, v in edges)


def compute_bandwidth(
    n: int,
    edges: List[Tuple[int, int]],
    permutation: List[int],
) -> int:
    """Return max |pi(u) - pi(v)| over all edges (u, v)."""
    if not edges:
        return 0
    return max(abs(permutation[u] - permutation[v]) for u, v in edges)


def validate_permutation(n: int, permutation: Optional[List[int]]) -> bool:
    """Return True iff permutation is a valid bijection of 0 .. n-1."""
    if permutation is None or len(permutation) != n:
        return False
    return sorted(permutation) == list(range(n))


def verify_solution(
    n: int,
    edges: List[Tuple[int, int]],
    result: Dict,
) -> Dict:
    """Validate a solver result and independently recompute bandwidth."""
    perm = result.get("permutation")
    valid = validate_permutation(n, perm)
    result["permutation_valid"] = valid

    if valid:
        bw = compute_bandwidth(n, edges, perm)
        result["verified_bandwidth"] = bw
        result["objective_match"] = (bw == result.get("bandwidth"))
    else:
        result["verified_bandwidth"] = None
        result["objective_match"] = None

    return result
