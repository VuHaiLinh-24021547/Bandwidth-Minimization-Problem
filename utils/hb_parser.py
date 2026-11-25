from __future__ import annotations

import io
import os
from typing import List, Tuple

try:
    from scipy import io as sp_io
    from scipy import sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _load_mtx_pure_python(filepath: str) -> Tuple[int, List[Tuple[int, int]]]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("%")]

    if lines and lines[0].lower().startswith("nombre"):
        lines = lines[1:]

    dim_idx = None
    n = 0
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            rows, cols = int(parts[0]), int(parts[1])
            if rows != cols:
                raise ValueError(f"Matrix is not square: {rows} rows × {cols} cols in '{filepath}'")
            n = rows
            dim_idx = idx
            break

    if dim_idx is None:
        raise ValueError(f"Could not find matrix dimensions in '{filepath}'")

    edges = set()
    for line in lines[dim_idx + 1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            if u != v and 0 <= u < n and 0 <= v < n:
                edges.add((u, v) if u < v else (v, u))

    return n, sorted(edges)


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
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    if not HAS_SCIPY:
        return _load_mtx_pure_python(filepath)

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
