from scipy.spatial import Delaunay
from scipy.spatial import QhullError

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import DelaunayInput, DelaunayResult
from nodes._common import (
    NodeInputError,
    array_to_int_matrix,
    check_matrix_shape,
    err,
    matrix_to_array,
)

MAX_POINTS = 20_000
ALLOWED_DIMS = (2, 3)


def delaunay_triangulate(ax: AxiomContext, input: DelaunayInput) -> DelaunayResult:
    """Compute the Delaunay triangulation of a 2D or 3D point set via
    scipy.spatial.Delaunay (Qhull). Returns each simplex as D+1 point
    indices (a triangle for 2D, a tetrahedron for 3D) plus the per-simplex
    neighbor structure (-1 where a facet has no neighbor, i.e. lies on the
    convex hull boundary). Points must be 2D or 3D; fewer than D+1 points
    returns INSUFFICIENT_POINTS; collinear/coplanar points that admit no
    triangulation return DEGENERATE_INPUT rather than crashing.
    """
    try:
        check_matrix_shape(input.points, max_rows=MAX_POINTS, max_cols=3)
        points = matrix_to_array(input.points)
        n, d = points.shape
        if d not in ALLOWED_DIMS:
            raise NodeInputError("INVALID_ARGUMENT", f"points must be 2D or 3D, got D={d}")
        if n < d + 1:
            raise NodeInputError(
                "INSUFFICIENT_POINTS", f"need at least {d + 1} points for a {d}D triangulation, got {n}"
            )

        try:
            tri = Delaunay(points)
        except QhullError as e:
            raise NodeInputError(
                "DEGENERATE_INPUT",
                f"points are degenerate (collinear/coplanar) for a {d}D triangulation: {e}",
            )

        return DelaunayResult(
            simplices=array_to_int_matrix(tri.simplices),
            neighbors=array_to_int_matrix(tri.neighbors),
        )
    except NodeInputError as e:
        return DelaunayResult(error=e.error)
    except Exception as e:
        return DelaunayResult(error=err("TRIANGULATION_FAILED", str(e)))
