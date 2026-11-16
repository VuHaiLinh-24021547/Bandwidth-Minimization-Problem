"""
hb_parser.py
============
Loads sparse-matrix files (Matrix Market, Harwell-Boeing) and extracts graph
properties for the Matrix Bandwidth Minimization Problem (BMP).

Supported formats:
- Matrix Market (.mtx, .mtx.rnd, or files beginning with %%MatrixMarket)
  loaded via scipy.io.mmread.
- Harwell-Boeing (.hb, .rua, .rsa, .psa) loaded via scipy.io.hb_read.
"""

from __future__ import annotations

import io
import os
import random
from typing import List, Tuple

import numpy as np
from scipy import io as sp_io
from scipy import sparse


def _read_raw_matrix(filepath: str) -> sparse.spmatrix:
    """Read a matrix from filepath using scipy.io.mmread or scipy.io.hb_read."""
    is_mm = filepath.endswith(".mtx.rnd") or filepath.endswith(".mtx")
    if not is_mm:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                if first_line.startswith("%%MatrixMarket"):
                    is_mm = True
        except Exception:
            pass

    if is_mm:
        try:
            return sp_io.mmread(filepath)
        except Exception:
            # Handle files that have custom headers like "Nombre del problema:"
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if lines and lines[0].strip().lower().startswith("nombre"):
                lines = lines[1:]

            # Check if lines have 2 or 3 tokens to determine coordinate type
            header = "%%MatrixMarket matrix coordinate pattern general\n"
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    header = "%%MatrixMarket matrix coordinate real general\n"
                    break
                elif len(parts) == 2:
                    header = "%%MatrixMarket matrix coordinate pattern general\n"
                    break

            return sp_io.mmread(io.StringIO(header + "".join(lines)))
    else:
        return sp_io.hb_read(filepath)


def load_hb_file(filepath: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Load a sparse-matrix file and convert to an undirected simple graph.

    Parameters
    ----------
    filepath : str
        Path to a Matrix Market (.mtx, .mtx.rnd) or Harwell-Boeing file.

    Returns
    -------
    n : int
        Number of vertices (0-indexed).
    edges : list of (int, int)
        Sorted list of unique undirected edges (u, v) with u < v.

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If the matrix is not square.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    raw_mat = _read_raw_matrix(filepath)
    A = sparse.csr_matrix(raw_mat)

    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Matrix is not square: {A.shape[0]} rows × {A.shape[1]} cols in '{filepath}'"
        )
    n = A.shape[0]

    # Symmetrize via S = |A| + |A|^T
    A_abs = abs(A)
    S = (A_abs + A_abs.T).tocsr()

    # Zero out the main diagonal (ignore self-loops)
    S.setdiag(0)
    S.eliminate_zeros()

    # Extract sorted unique undirected edges E = {(u, v) | S_{u,v} != 0 and u < v}
    coo = sparse.triu(S, k=1, format="coo")
    edges = sorted(set(zip(coo.row.tolist(), coo.col.tolist())))

    return n, edges


def generate_sample_hb(filepath: str, n: int = 15) -> str:
    """Create a synthetic connected test file and return its path."""
    random.seed(42)
    edges: set[Tuple[int, int]] = set()

    # Ensure connectivity with a random spanning tree
    vertices = list(range(n))
    random.shuffle(vertices)
    for i in range(1, n):
        u, v = vertices[i - 1], vertices[i]
        edges.add((min(u, v), max(u, v)))

    # Add extra random edges to bring the count up to ~2n
    target = min(2 * n, n * (n - 1) // 2)
    attempts = 0
    while len(edges) < target and attempts < 10 * target:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
        attempts += 1

    sorted_edges = sorted(edges)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(f"Nombre del problema: synthetic_{n}\n")
        fh.write(f"{n} {n} {len(sorted_edges)}\n")
        for u, v in sorted_edges:
            fh.write(f"{u + 1} {v + 1}\n")  # 1-based in file

    return os.path.abspath(filepath)
