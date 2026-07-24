from scipy.spatial.distance import pdist

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import PdistInput, PdistResult
from nodes._common import (
    NodeInputError,
    array_to_vector,
    check_matrix_shape,
    err,
    matrix_to_array,
    resolve_full_metric_kwargs,
)

def pairwise_distance_matrix(ax: AxiomContext, input: PdistInput) -> PdistResult:
    """Compute the condensed (upper-triangular) pairwise distance vector
    among all rows of `points`, via scipy.spatial.distance.pdist. metric is
    "euclidean" (default), "cityblock", "cosine", "chebyshev", "minkowski"
    (uses `p`), "hamming", "jaccard", or "mahalanobis" (uses `inv_cov` if
    supplied, else scipy's own internally-derived sample
    inverse-covariance). Returns the condensed vector (length n*(n-1)/2)
    plus `n` so the caller can index or squareform-expand it.
    """
    try:
        check_matrix_shape(input.points, min_rows=2, name="points")
        points = matrix_to_array(input.points)
        n = points.shape[0]

        metric, kwargs = resolve_full_metric_kwargs(
            input.metric, input.p, input.inv_cov if input.HasField("inv_cov") else None,
            require_inv_cov_for_mahalanobis=False,
        )

        condensed = pdist(points, metric=metric, **kwargs)

        return PdistResult(condensed=array_to_vector(condensed), n=n)
    except NodeInputError as e:
        return PdistResult(error=e.error)
    except Exception as e:
        return PdistResult(error=err("DISTANCE_FAILED", str(e)))
