"""Shared helpers for christiangeorgelucas/spatial-tools nodes.

Every node consumes/produces the shared Matrix/Vector/IntVector/IntMatrix/
BoolVector envelopes defined in messages.proto (the same row-major Matrix
convention used by christiangeorgelucas/numerics-tools and
christiangeorgelucas/sklearn-tools — see messages.proto's header comment).
This module centralizes:
  - proto <-> numpy conversions
  - input-size guards, enforced BEFORE allocating a numpy array from the
    caller-supplied Matrix so a claimed huge dimension is rejected on the
    proto's own row/col counts, not after materializing a huge array
  - the shared structured-error contract (Error{code, message})

Every node in this package is fully deterministic (scipy.spatial/scipy.
spatial.distance/numpy have no randomness in any op this package wraps) —
there is no seed/RNG plumbing to carry, unlike a package that fits a model.
"""
from __future__ import annotations

import numpy as np

from gen.messages_pb2 import BoolVector, Error, IntMatrix, IntVector, Matrix, Vector

# Generic per-point-set caps used as a default by the cheap, non-combinatorial
# nodes (BoundingBox). Nodes with a combinatorially expensive underlying
# algorithm (Delaunay/Voronoi/ConvexHull/Pdist/Hausdorff/...) apply their own,
# tighter caps at the call site — see each node's axiom.yaml description for
# the exact number and rationale.
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_COLS = 50

# Minkowski-family metrics scipy.spatial.cKDTree natively accelerates.
KDTREE_METRICS = {"euclidean": 2, "cityblock": 1, "chebyshev": np.inf}

# Full scipy.spatial.distance metric vocabulary this package exposes on the
# non-KD-tree-accelerated distance nodes (Pdist/Cdist/PointDistance).
FULL_METRICS = {
    "euclidean",
    "cityblock",
    "cosine",
    "chebyshev",
    "minkowski",
    "hamming",
    "jaccard",
    "mahalanobis",
}


def err(code: str, message: str) -> Error:
    return Error(code=code, message=message)


class NodeInputError(Exception):
    """Raised by helpers to short-circuit a node with a structured Error.
    Every node catches this at the top level and returns it as `error`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.error = err(code, message)


def check_matrix_shape(
    matrix: Matrix,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
    min_rows: int = 1,
    name: str = "points",
) -> None:
    """Cheap, allocation-free shape validation, run before converting to
    numpy. Raises NodeInputError on any problem."""
    n_rows = len(matrix.rows)
    if n_rows < min_rows:
        raise NodeInputError(
            "EMPTY_INPUT", f"{name} must have at least {min_rows} row(s), got {n_rows}"
        )
    n_cols = len(matrix.rows[0].values)
    if n_cols == 0:
        raise NodeInputError("EMPTY_INPUT", f"{name} rows must have at least one column")
    for i, row in enumerate(matrix.rows):
        if len(row.values) != n_cols:
            raise NodeInputError(
                "RAGGED_MATRIX",
                f"{name} row {i} has {len(row.values)} columns, expected {n_cols}",
            )
    if n_rows > max_rows or n_cols > max_cols:
        raise NodeInputError(
            "TOO_LARGE", f"{name} is {n_rows}x{n_cols}; max is {max_rows}x{max_cols}"
        )


def matrix_to_array(matrix: Matrix, *, allow_nonfinite: bool = False, name: str = "points") -> np.ndarray:
    """Convert an already shape-checked Matrix to a numpy float64 2D array."""
    arr = np.array([list(row.values) for row in matrix.rows], dtype=np.float64)
    if not allow_nonfinite and not np.all(np.isfinite(arr)):
        raise NodeInputError("NON_FINITE", f"{name} must not contain NaN or Infinity")
    return arr


def array_to_matrix(arr) -> Matrix:
    arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    m = Matrix()
    for row in arr:
        r = m.rows.add()
        r.values.extend(row.tolist())
    return m


def vector_to_array(vector: Vector, *, allow_nonfinite: bool = False, name: str = "vector") -> np.ndarray:
    arr = np.array(list(vector.values), dtype=np.float64)
    if not allow_nonfinite and not np.all(np.isfinite(arr)):
        raise NodeInputError("NON_FINITE", f"{name} must not contain NaN or Infinity")
    return arr


def array_to_vector(arr) -> Vector:
    return Vector(values=[float(v) for v in np.asarray(arr, dtype=np.float64).ravel()])


def int_vector_to_array(vector: IntVector) -> np.ndarray:
    return np.array(list(vector.values), dtype=np.int64)


def array_to_int_vector(arr) -> IntVector:
    return IntVector(values=[int(v) for v in np.asarray(arr).ravel()])


def array_to_int_matrix(arr) -> IntMatrix:
    m = IntMatrix()
    for row in np.atleast_2d(np.asarray(arr)):
        r = m.rows.add()
        r.values.extend(int(v) for v in row)
    return m


def ragged_to_int_matrix(rows) -> IntMatrix:
    """Build an IntMatrix from a list of variable-length int sequences
    (ragged rows) — used by nodes whose result rows legitimately differ in
    length (RadiusNeighbors, QueryPairs' -is not ragged but Voronoi
    regions/ridge_vertices are)."""
    m = IntMatrix()
    for row in rows:
        r = m.rows.add()
        r.values.extend(int(v) for v in row)
    return m


def array_to_bool_vector(arr) -> BoolVector:
    return BoolVector(values=[bool(v) for v in np.asarray(arr).ravel()])


def check_output_cells(n_rows: int, n_cols: int, *, cap: int, label: str = "output") -> None:
    """Guard an output matrix's size BEFORE computing it, for nodes whose
    output shape is a function of caller-supplied parameters (k, ...) rather
    than a direct echo of the input's own shape."""
    if n_rows < 0 or n_cols < 0:
        raise NodeInputError("INVALID_ARGUMENT", f"{label} has a negative dimension")
    if n_rows * n_cols > cap:
        raise NodeInputError(
            "TOO_LARGE",
            f"{label} would be {n_rows}x{n_cols} ({n_rows * n_cols} cells); max is {cap} cells",
        )


def check_result_count(count: int, *, cap: int, label: str = "result") -> None:
    """Guard a result COUNT computed post-hoc (query_pairs, query_ball_point
    total neighbors) — these can't be bounded before running the query, so
    they're checked immediately after and before serializing the response."""
    if count > cap:
        raise NodeInputError(
            "TOO_LARGE",
            f"{label} has {count} entries; max is {cap} — narrow the radius/distance or input size",
        )


def resolve_kdtree_metric(metric: str) -> tuple[str, float]:
    """Validate a KD-tree-node metric string and return (name, p) for
    scipy.spatial.cKDTree's `p` parameter."""
    metric = metric or "euclidean"
    if metric not in KDTREE_METRICS:
        raise NodeInputError(
            "INVALID_ARGUMENT",
            f"metric must be one of {sorted(KDTREE_METRICS)} for a KD-tree-accelerated node "
            "(use PairwiseDistanceMatrix/CrossDistanceMatrix for cosine/hamming/jaccard/mahalanobis)",
        )
    return metric, KDTREE_METRICS[metric]


def resolve_full_metric_kwargs(metric: str, p: float, inv_cov, *, require_inv_cov_for_mahalanobis: bool) -> tuple[str, dict]:
    """Validate a Pdist/Cdist/PointDistance metric string and build the
    scipy.spatial.distance kwargs dict for it."""
    metric = metric or "euclidean"
    if metric not in FULL_METRICS:
        raise NodeInputError("INVALID_ARGUMENT", f"metric must be one of {sorted(FULL_METRICS)}")
    kwargs: dict = {}
    if metric == "minkowski":
        kwargs["p"] = p if p and p > 0 else 2.0
    if metric == "mahalanobis":
        if inv_cov is not None and len(inv_cov.rows) > 0:
            kwargs["VI"] = matrix_to_array(inv_cov, name="inv_cov")
        elif require_inv_cov_for_mahalanobis:
            raise NodeInputError(
                "INVALID_ARGUMENT",
                "metric=mahalanobis requires inv_cov to be supplied explicitly here "
                "(no sample to derive a covariance from two points)",
            )
        # else: leave VI unset — scipy derives it internally from the sample.
    return metric, kwargs
