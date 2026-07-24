import numpy as np
from scipy.spatial import ConvexHull, QhullError

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import PointInHullInput, PointInHullResult
from nodes._common import (
    NodeInputError,
    array_to_bool_vector,
    check_matrix_shape,
    err,
    matrix_to_array,
)

MIN_DIM = 2
DEFAULT_TOL = 1e-9


def point_in_convex_hull(ax: AxiomContext, input: PointInHullInput) -> PointInHullResult:
    """Test whether each point in `queries` lies inside (or on the boundary
    of, within `tol`) the convex hull of `points`, via
    scipy.spatial.ConvexHull's facet equations (hull.equations) — the
    standard scipy-documented point-in-hull technique. `tol` defaults to
    1e-9 when <= 0.
    """
    try:
        check_matrix_shape(input.points, name="points")
        check_matrix_shape(input.queries, name="queries")
        points = matrix_to_array(input.points)
        queries = matrix_to_array(input.queries)
        n, d = points.shape
        if d < MIN_DIM:
            raise NodeInputError("INVALID_ARGUMENT", f"points must have D >= {MIN_DIM}, got D={d}")
        if points.shape[1] != queries.shape[1]:
            raise NodeInputError(
                "DIMENSION_MISMATCH",
                f"points has {points.shape[1]} columns, queries has {queries.shape[1]}",
            )
        if n < d + 1:
            raise NodeInputError(
                "INSUFFICIENT_POINTS", f"need at least {d + 1} points for a {d}D hull, got {n}"
            )

        try:
            hull = ConvexHull(points)
        except QhullError as e:
            raise NodeInputError(
                "DEGENERATE_INPUT", f"points are degenerate (coplanar/collinear, rank < {d}): {e}"
            )

        tol = input.tol if input.tol > 0 else DEFAULT_TOL
        # hull.equations: (n_facets, D+1) rows [normal..., offset]; a point
        # is inside/on-boundary iff normal . p + offset <= tol for every facet.
        eq = hull.equations
        inside = np.all(eq[:, :-1] @ queries.T + eq[:, -1:] <= tol, axis=0)

        return PointInHullResult(inside=array_to_bool_vector(inside))
    except NodeInputError as e:
        return PointInHullResult(error=e.error)
    except Exception as e:
        return PointInHullResult(error=err("HULL_TEST_FAILED", str(e)))
