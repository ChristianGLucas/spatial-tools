from scipy.spatial import cKDTree

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import RadiusNeighborsInput, RadiusNeighborsResult
from nodes._common import (
    NodeInputError,
    check_matrix_shape,
    check_result_count,
    err,
    matrix_to_array,
    ragged_to_int_matrix,
    resolve_kdtree_metric,
)

MAX_POINTS = 20_000
MAX_QUERIES = 20_000
MAX_DIM = 64
MAX_TOTAL_NEIGHBORS = 200_000


def radius_neighbors(ax: AxiomContext, input: RadiusNeighborsInput) -> RadiusNeighborsResult:
    """Build a KD-tree over `points` and, for each row in `queries`, return
    every point within `radius` (a "ball query"), via
    scipy.spatial.cKDTree.query_ball_point. metric is "euclidean" (default),
    "cityblock", or "chebyshev". Result rows are ragged (variable neighbor
    count per query) and not guaranteed sorted by distance.
    """
    try:
        check_matrix_shape(input.points, max_rows=MAX_POINTS, max_cols=MAX_DIM, name="points")
        check_matrix_shape(input.queries, max_rows=MAX_QUERIES, max_cols=MAX_DIM, name="queries")
        points = matrix_to_array(input.points)
        queries = matrix_to_array(input.queries)
        if points.shape[1] != queries.shape[1]:
            raise NodeInputError(
                "DIMENSION_MISMATCH",
                f"points has {points.shape[1]} columns, queries has {queries.shape[1]}",
            )
        if input.radius <= 0:
            raise NodeInputError("INVALID_ARGUMENT", f"radius must be > 0 (got {input.radius})")

        metric, p = resolve_kdtree_metric(input.metric)

        tree = cKDTree(points)
        neighbor_lists = tree.query_ball_point(queries, r=input.radius, p=p)

        total = sum(len(row) for row in neighbor_lists)
        check_result_count(total, cap=MAX_TOTAL_NEIGHBORS, label="total neighbors across all queries")

        return RadiusNeighborsResult(indices=ragged_to_int_matrix(neighbor_lists))
    except NodeInputError as e:
        return RadiusNeighborsResult(error=e.error)
    except Exception as e:
        return RadiusNeighborsResult(error=err("QUERY_FAILED", str(e)))
