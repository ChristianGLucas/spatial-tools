import numpy as np
from scipy.spatial.distance import pdist

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import PointDistanceInput, PointDistanceResult
from nodes._common import (
    NodeInputError,
    err,
    resolve_full_metric_kwargs,
    vector_to_array,
)

def point_distance(ax: AxiomContext, input: PointDistanceInput) -> PointDistanceResult:
    """Compute the distance between exactly two D-dimensional points under
    a chosen metric, via scipy.spatial.distance. metric is "euclidean"
    (default), "cityblock", "cosine", "chebyshev", "minkowski" (uses `p`),
    "hamming", "jaccard", or "mahalanobis" (requires `inv_cov` to be
    supplied explicitly — with only two points there is no sample to
    derive a covariance from). `a` and `b` must have equal length.
    """
    try:
        a = vector_to_array(input.a, name="a")
        b = vector_to_array(input.b, name="b")
        if len(a) == 0 or len(b) == 0:
            raise NodeInputError("EMPTY_INPUT", "a and b must each have at least one dimension")
        if len(a) != len(b):
            raise NodeInputError("DIMENSION_MISMATCH", f"a has {len(a)} dims, b has {len(b)} dims")

        metric, kwargs = resolve_full_metric_kwargs(
            input.metric, input.p, input.inv_cov if input.HasField("inv_cov") else None,
            require_inv_cov_for_mahalanobis=True,
        )

        # Route through pdist on the 2-row stack so this node's numeric
        # behavior is exactly consistent with Pdist/Cdist for the same
        # metric (rather than a separately hand-rolled formula).
        distance = float(pdist(np.vstack([a, b]), metric=metric, **kwargs)[0])

        return PointDistanceResult(distance=distance)
    except NodeInputError as e:
        return PointDistanceResult(error=e.error)
    except Exception as e:
        return PointDistanceResult(error=err("DISTANCE_FAILED", str(e)))
