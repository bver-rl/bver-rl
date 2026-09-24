# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height fills for the un-scanned cells of a scan's bounding box, free of Isaac Lab imports.

``fill_harmonic`` (default) blends the whole rim without steps and never exceeds it; ``fill_nearest``
extrudes the nearest rim cell and can leave flat shelves.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fill_harmonic", "fill_nearest"]


def fill_nearest(heights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells from their nearest valid neighbour."""
    if valid.all():
        return heights
    from scipy import ndimage

    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    return heights[tuple(indices)]


def fill_harmonic(heights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells with the harmonic interpolant of the valid rim.

    Five-point Laplacian over the invalid cells, Dirichlet on valid ones, Neumann at the array border.
    """
    if valid.all():
        return heights
    assert valid.any(), "cannot fill a grid with no valid cells"
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    hole = ~valid
    n = int(hole.sum())
    nx, ny = valid.shape
    index = np.full(valid.shape, -1, dtype=np.int64)
    index[hole] = np.arange(n)
    ii, jj = np.nonzero(hole)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs = np.zeros(n)
    diag = np.zeros(n)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        inb = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        diag += inb  # border cells simply have fewer neighbours
        nb_hole = np.zeros(n, dtype=bool)
        nb_hole[inb] = hole[ni[inb], nj[inb]]
        unknown = inb & nb_hole
        rows.append(index[ii[unknown], jj[unknown]])
        cols.append(index[ni[unknown], nj[unknown]])
        vals.append(-np.ones(int(unknown.sum())))
        dirichlet = inb & ~nb_hole
        np.add.at(rhs, index[ii[dirichlet], jj[dirichlet]], heights[ni[dirichlet], nj[dirichlet]].astype(np.float64))
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    matrix = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    filled = np.array(heights, dtype=np.float64, copy=True)
    filled[hole] = spsolve(matrix, rhs)
    return filled.astype(heights.dtype)
