from scipy.spatial import procrustes

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import ProcrustesInput, ProcrustesResult
from nodes._common import (
    NodeInputError,
    array_to_matrix,
    check_matrix_shape,
    err,
    matrix_to_array,
)

MAX_CELLS = 350_000


def procrustes_disparity(ax: AxiomContext, input: ProcrustesInput) -> ProcrustesResult:
    """Compute the Procrustes disparity between two equally-shaped point
    sets (shape alignment), via scipy.spatial.procrustes — standardizes
    both point sets and finds the optimal rotation of `points_b` onto
    `points_a`, returning the sum-of-squared-errors disparity (0 =
    identical shape) plus both standardized/aligned point sets.
    """
    try:
        check_matrix_shape(input.points_a, max_rows=100_000, max_cols=256, min_rows=2, name="points_a")
        check_matrix_shape(input.points_b, max_rows=100_000, max_cols=256, min_rows=2, name="points_b")
        a = matrix_to_array(input.points_a, name="points_a")
        b = matrix_to_array(input.points_b, name="points_b")

        if a.shape != b.shape:
            raise NodeInputError(
                "DIMENSION_MISMATCH", f"points_a has shape {a.shape}, points_b has shape {b.shape}"
            )
        if a.shape[0] * a.shape[1] > MAX_CELLS:
            raise NodeInputError(
                "TOO_LARGE", f"points_a/points_b have {a.shape[0] * a.shape[1]} cells each; max is {MAX_CELLS}"
            )

        try:
            aligned_a, aligned_b, disparity = procrustes(a, b)
        except ValueError as e:
            raise NodeInputError("DEGENERATE_INPUT", str(e))

        return ProcrustesResult(
            disparity=float(disparity),
            aligned_a=array_to_matrix(aligned_a),
            aligned_b=array_to_matrix(aligned_b),
        )
    except NodeInputError as e:
        return ProcrustesResult(error=e.error)
    except Exception as e:
        return ProcrustesResult(error=err("PROCRUSTES_FAILED", str(e)))
