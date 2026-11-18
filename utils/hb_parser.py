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
    """Load a sparse-matrix file and convert to an undirected simple graph."""
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
