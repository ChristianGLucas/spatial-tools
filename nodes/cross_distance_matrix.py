from scipy.spatial.distance import cdist

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import CdistInput, CdistResult
from nodes._common import (
    NodeInputError,
    array_to_matrix,
    check_matrix_shape,
    err,
    matrix_to_array,
    resolve_full_metric_kwargs,
)


def cross_distance_matrix(ax: AxiomContext, input: CdistInput) -> CdistResult:
    """Compute the full cross distance matrix between two point sets
    `points_a` (N x D) and `points_b` (M x D), via
    scipy.spatial.distance.cdist. Same metric/p/inv_cov options as
    PairwiseDistanceMatrix. Returns an N x M matrix,
    distances[i][j] = distance(points_a[i], points_b[j]).
    """
    try:
        check_matrix_shape(input.points_a, name="points_a")
        check_matrix_shape(input.points_b, name="points_b")
        a = matrix_to_array(input.points_a, name="points_a")
        b = matrix_to_array(input.points_b, name="points_b")
        if a.shape[1] != b.shape[1]:
            raise NodeInputError(
                "DIMENSION_MISMATCH", f"points_a has {a.shape[1]} columns, points_b has {b.shape[1]}"
            )

        metric, kwargs = resolve_full_metric_kwargs(
            input.metric, input.p, input.inv_cov if input.HasField("inv_cov") else None,
            require_inv_cov_for_mahalanobis=False,
        )

        distances = cdist(a, b, metric=metric, **kwargs)

        return CdistResult(distances=array_to_matrix(distances))
    except NodeInputError as e:
        return CdistResult(error=e.error)
    except Exception as e:
        return CdistResult(error=err("DISTANCE_FAILED", str(e)))
