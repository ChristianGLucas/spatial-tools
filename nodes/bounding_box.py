from gen.axiom_context import AxiomContext
from gen.messages_pb2 import BoundingBoxInput, BoundingBoxResult
from nodes._common import NodeInputError, array_to_vector, check_matrix_shape, err, matrix_to_array

def bounding_box(ax: AxiomContext, input: BoundingBoxInput) -> BoundingBoxResult:
    """Compute the axis-aligned bounding box of a point set: the
    per-dimension minimum and maximum across all rows of `points`, via
    numpy.min/numpy.max.
    """
    try:
        check_matrix_shape(input.points, name="points")
        points = matrix_to_array(input.points)

        return BoundingBoxResult(
            min=array_to_vector(points.min(axis=0)), max=array_to_vector(points.max(axis=0))
        )
    except NodeInputError as e:
        return BoundingBoxResult(error=e.error)
    except Exception as e:
        return BoundingBoxResult(error=err("BOUNDING_BOX_FAILED", str(e)))
