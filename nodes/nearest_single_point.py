from scipy.spatial import cKDTree

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import NearestSinglePointInput, NearestSinglePointResult
from nodes._common import (
    NodeInputError,
    check_matrix_shape,
    err,
    matrix_to_array,
    resolve_kdtree_metric,
    vector_to_array,
)

def nearest_single_point(ax: AxiomContext, input: NearestSinglePointInput) -> NearestSinglePointResult:
    """Find the single nearest point in `points` to one `query` point, via
    scipy.spatial.cKDTree — a convenience node for the common single-query
    case (see KNearestNeighbors for the batched-queries, k>=1 general case).
    metric is "euclidean" (default), "cityblock", or "chebyshev".
    """
    try:
        check_matrix_shape(input.points, name="points")
        points = matrix_to_array(input.points)
        query = vector_to_array(input.query, name="query")
        if len(query) == 0:
            raise NodeInputError("EMPTY_INPUT", "query must have at least one dimension")
        if points.shape[1] != len(query):
            raise NodeInputError(
                "DIMENSION_MISMATCH",
                f"points has {points.shape[1]} columns, query has {len(query)} dimensions",
            )

        metric, p = resolve_kdtree_metric(input.metric)

        tree = cKDTree(points)
        dist, idx = tree.query(query, k=1, p=p)

        return NearestSinglePointResult(index=int(idx), distance=float(dist))
    except NodeInputError as e:
        return NearestSinglePointResult(error=e.error)
    except Exception as e:
        return NearestSinglePointResult(error=err("QUERY_FAILED", str(e)))
