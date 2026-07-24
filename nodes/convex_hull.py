from scipy.spatial import ConvexHull as ScipyConvexHull
from scipy.spatial import QhullError

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import ConvexHullInput, ConvexHullResult
from nodes._common import (
    NodeInputError,
    array_to_int_matrix,
    array_to_int_vector,
    check_matrix_shape,
    err,
    matrix_to_array,
)

MIN_DIM = 2


def convex_hull(ax: AxiomContext, input: ConvexHullInput) -> ConvexHullResult:
    """Compute the convex hull of a general N-dimensional point set (D >= 2)
    via scipy.spatial.ConvexHull (Qhull). Returns the hull's
    extreme-point (vertex) indices, its facet simplices, its D-dimensional
    volume (area, for D=2), and its (D-1)-dimensional surface measure
    (perimeter, for D=2). Fewer than D+1 points returns
    INSUFFICIENT_POINTS; coplanar/collinear (rank-deficient) points return
    DEGENERATE_INPUT.
    """
    try:
        check_matrix_shape(input.points)
        points = matrix_to_array(input.points)
        n, d = points.shape
        if d < MIN_DIM:
            raise NodeInputError("INVALID_ARGUMENT", f"points must have D >= {MIN_DIM}, got D={d}")
        if n < d + 1:
            raise NodeInputError(
                "INSUFFICIENT_POINTS", f"need at least {d + 1} points for a {d}D hull, got {n}"
            )

        try:
            hull = ScipyConvexHull(points)
        except QhullError as e:
            raise NodeInputError(
                "DEGENERATE_INPUT", f"points are degenerate (coplanar/collinear, rank < {d}): {e}"
            )

        return ConvexHullResult(
            vertices=array_to_int_vector(hull.vertices),
            simplices=array_to_int_matrix(hull.simplices),
            volume=float(hull.volume),
            area=float(hull.area),
        )
    except NodeInputError as e:
        return ConvexHullResult(error=e.error)
    except Exception as e:
        return ConvexHullResult(error=err("HULL_FAILED", str(e)))
