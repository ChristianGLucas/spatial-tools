from scipy.spatial import cKDTree

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import QueryPairsInput, QueryPairsResult
from nodes._common import (
    NodeInputError,
    array_to_int_matrix,
    check_matrix_shape,
    check_result_count,
    err,
    matrix_to_array,
    resolve_kdtree_metric,
)

MAX_POINTS = 20_000
MAX_DIM = 64
MAX_PAIRS = 200_000


def query_pairs(ax: AxiomContext, input: QueryPairsInput) -> QueryPairsResult:
    """Find every pair of points within `distance` of each other, inside a
    single point set, via scipy.spatial.cKDTree.query_pairs. metric is
    "euclidean" (default), "cityblock", or "chebyshev". Returns each
    qualifying pair (i, j) with i < j as row indices into `points`.
    """
    try:
        check_matrix_shape(input.points, max_rows=MAX_POINTS, max_cols=MAX_DIM, name="points")
        points = matrix_to_array(input.points)
        if input.distance <= 0:
            raise NodeInputError("INVALID_ARGUMENT", f"distance must be > 0 (got {input.distance})")

        metric, p = resolve_kdtree_metric(input.metric)

        tree = cKDTree(points)
        pairs = tree.query_pairs(r=input.distance, p=p, output_type="ndarray")

        check_result_count(len(pairs), cap=MAX_PAIRS, label="pairs")

        return QueryPairsResult(pairs=array_to_int_matrix(pairs))
    except NodeInputError as e:
        return QueryPairsResult(error=e.error)
    except Exception as e:
        return QueryPairsResult(error=err("QUERY_FAILED", str(e)))
