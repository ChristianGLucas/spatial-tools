import numpy as np
from scipy.spatial import cKDTree

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import KNNInput, KNNResult
from nodes._common import (
    NodeInputError,
    array_to_int_matrix,
    array_to_matrix,
    check_matrix_shape,
    err,
    matrix_to_array,
    resolve_kdtree_metric,
)


def k_nearest_neighbors(ax: AxiomContext, input: KNNInput) -> KNNResult:
    """Build a KD-tree over `points` (scipy.spatial.cKDTree) and query the
    k nearest neighbors of each row in `queries`. metric is "euclidean"
    (default), "cityblock", or "chebyshev". Returns Q x k index and
    distance matrices in ascending-distance order per row. k is rejected
    with INVALID_ARGUMENT if it exceeds len(points) — every returned row is
    always fully populated with k real neighbors, never padded.
    """
    try:
        check_matrix_shape(input.points, name="points")
        check_matrix_shape(input.queries, name="queries")
        points = matrix_to_array(input.points)
        queries = matrix_to_array(input.queries)
        if points.shape[1] != queries.shape[1]:
            raise NodeInputError(
                "DIMENSION_MISMATCH",
                f"points has {points.shape[1]} columns, queries has {queries.shape[1]}",
            )

        k = input.k
        if k < 1 or k > points.shape[0]:
            raise NodeInputError(
                "INVALID_ARGUMENT",
                f"k must satisfy 1 <= k <= len(points) = {points.shape[0]} (got {k})",
            )

        metric, p = resolve_kdtree_metric(input.metric)

        tree = cKDTree(points)
        dist, idx = tree.query(queries, k=k, p=p)
        dist = np.atleast_2d(dist.reshape(queries.shape[0], k))
        idx = np.atleast_2d(idx.reshape(queries.shape[0], k)).astype(np.int64)
        idx[np.isinf(dist)] = -1

        return KNNResult(indices=array_to_int_matrix(idx), distances=array_to_matrix(dist))
    except NodeInputError as e:
        return KNNResult(error=e.error)
    except Exception as e:
        return KNNResult(error=err("QUERY_FAILED", str(e)))
