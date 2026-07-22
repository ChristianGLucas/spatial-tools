from scipy.spatial.distance import directed_hausdorff

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import HausdorffInput, HausdorffResult
from nodes._common import (
    NodeInputError,
    array_to_int_vector,
    check_matrix_shape,
    err,
    matrix_to_array,
)

MAX_POINTS = 20_000
MAX_DIM = 256


def hausdorff_distance(ax: AxiomContext, input: HausdorffInput) -> HausdorffResult:
    """Compute the symmetric Hausdorff distance between two point sets
    `points_a` and `points_b`, via
    max(scipy.spatial.distance.directed_hausdorff(A,B), directed_hausdorff(B,A)).
    Returns the distance, which direction achieved it (from_a), and the
    achieving index pair.
    """
    try:
        check_matrix_shape(input.points_a, max_rows=MAX_POINTS, max_cols=MAX_DIM, name="points_a")
        check_matrix_shape(input.points_b, max_rows=MAX_POINTS, max_cols=MAX_DIM, name="points_b")
        a = matrix_to_array(input.points_a, name="points_a")
        b = matrix_to_array(input.points_b, name="points_b")
        if a.shape[1] != b.shape[1]:
            raise NodeInputError(
                "DIMENSION_MISMATCH", f"points_a has {a.shape[1]} columns, points_b has {b.shape[1]}"
            )

        d_ab, i_a, j_b = directed_hausdorff(a, b)
        d_ba, i_b, j_a = directed_hausdorff(b, a)

        if d_ab >= d_ba:
            return HausdorffResult(
                distance=float(d_ab), from_a=True, index_pair=array_to_int_vector([i_a, j_b])
            )
        return HausdorffResult(
            distance=float(d_ba), from_a=False, index_pair=array_to_int_vector([i_b, j_a])
        )
    except NodeInputError as e:
        return HausdorffResult(error=e.error)
    except Exception as e:
        return HausdorffResult(error=err("DISTANCE_FAILED", str(e)))
