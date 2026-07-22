from scipy.spatial import QhullError, Voronoi

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import VoronoiInput, VoronoiResult
from nodes._common import (
    NodeInputError,
    array_to_int_vector,
    array_to_matrix,
    check_matrix_shape,
    err,
    matrix_to_array,
    ragged_to_int_matrix,
)

MAX_POINTS = 20_000
ALLOWED_DIMS = (2, 3)


def voronoi_diagram(ax: AxiomContext, input: VoronoiInput) -> VoronoiResult:
    """Compute the Voronoi diagram of a 2D or 3D point set via
    scipy.spatial.Voronoi (Qhull) — the geometric dual of Delaunay
    triangulation. Returns the finite Voronoi vertices, each input point's
    (possibly unbounded, ragged) region as a list of vertex indices with -1
    marking an unbounded direction, the point_region mapping from input
    point to its region, and the ridge structure.
    """
    try:
        check_matrix_shape(input.points, max_rows=MAX_POINTS, max_cols=3)
        points = matrix_to_array(input.points)
        n, d = points.shape
        if d not in ALLOWED_DIMS:
            raise NodeInputError("INVALID_ARGUMENT", f"points must be 2D or 3D, got D={d}")
        if n < d + 1:
            raise NodeInputError(
                "INSUFFICIENT_POINTS", f"need at least {d + 1} points for a {d}D Voronoi diagram, got {n}"
            )

        try:
            vor = Voronoi(points)
        except QhullError as e:
            raise NodeInputError(
                "DEGENERATE_INPUT",
                f"points are degenerate (collinear/coplanar) for a {d}D Voronoi diagram: {e}",
            )

        return VoronoiResult(
            vertices=array_to_matrix(vor.vertices),
            regions=ragged_to_int_matrix(vor.regions),
            point_region=array_to_int_vector(vor.point_region),
            ridge_points=ragged_to_int_matrix(vor.ridge_points),
            ridge_vertices=ragged_to_int_matrix(vor.ridge_vertices),
        )
    except NodeInputError as e:
        return VoronoiResult(error=e.error)
    except Exception as e:
        return VoronoiResult(error=err("VORONOI_FAILED", str(e)))
