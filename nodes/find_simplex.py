import numpy as np
from scipy.spatial import Delaunay, QhullError

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import FindSimplexInput, FindSimplexResult
from nodes._common import (
    NodeInputError,
    array_to_int_vector,
    array_to_matrix,
    check_matrix_shape,
    err,
    matrix_to_array,
)

MAX_POINTS = 20_000
MAX_QUERIES = 20_000
ALLOWED_DIMS = (2, 3)


def find_simplex(ax: AxiomContext, input: FindSimplexInput) -> FindSimplexResult:
    """Locate which Delaunay simplex (of `points`' 2D/3D triangulation, via
    scipy.spatial.Delaunay.find_simplex) each row of `queries` falls into,
    and its barycentric weights within that simplex — set-up for
    barycentric interpolation. Returns -1 for a query point outside the
    triangulation's convex hull (with an all-zero barycentric row).
    """
    try:
        check_matrix_shape(input.points, max_rows=MAX_POINTS, max_cols=3, name="points")
        check_matrix_shape(input.queries, max_rows=MAX_QUERIES, max_cols=3, name="queries")
        points = matrix_to_array(input.points)
        queries = matrix_to_array(input.queries)
        n, d = points.shape
        if d not in ALLOWED_DIMS:
            raise NodeInputError("INVALID_ARGUMENT", f"points must be 2D or 3D, got D={d}")
        if points.shape[1] != queries.shape[1]:
            raise NodeInputError(
                "DIMENSION_MISMATCH",
                f"points has {points.shape[1]} columns, queries has {queries.shape[1]}",
            )
        if n < d + 1:
            raise NodeInputError(
                "INSUFFICIENT_POINTS", f"need at least {d + 1} points for a {d}D triangulation, got {n}"
            )

        try:
            tri = Delaunay(points)
        except QhullError as e:
            raise NodeInputError(
                "DEGENERATE_INPUT", f"points are degenerate (collinear/coplanar) for a {d}D triangulation: {e}"
            )

        simplex_idx = tri.find_simplex(queries)
        q = queries.shape[0]
        bary = np.zeros((q, d + 1), dtype=np.float64)

        valid = simplex_idx >= 0
        if np.any(valid):
            valid_simplex = simplex_idx[valid]
            X = tri.transform[valid_simplex, :d, :d]
            Y = queries[valid] - tri.transform[valid_simplex, d, :]
            b = np.einsum("ijk,ik->ij", X, Y)
            last = 1.0 - b.sum(axis=1, keepdims=True)
            bary[valid] = np.hstack([b, last])

        return FindSimplexResult(
            simplex_indices=array_to_int_vector(simplex_idx),
            barycentric_coords=array_to_matrix(bary),
        )
    except NodeInputError as e:
        return FindSimplexResult(error=e.error)
    except Exception as e:
        return FindSimplexResult(error=err("FIND_SIMPLEX_FAILED", str(e)))
