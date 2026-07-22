from gen.messages_pb2 import PointInHullInput
from nodes.point_in_convex_hull import point_in_convex_hull
from nodes._test_helpers import COLLINEAR_2D, UNIT_SQUARE, _TestContext, make_matrix


def test_inside_outside_and_boundary_points():
    ax = _TestContext()
    queries = [
        [0.5, 0.5],  # center: clearly inside
        [2.0, 2.0],  # clearly outside
        [0.0, 0.0],  # exactly a hull vertex: on the boundary, counts as inside
        [-0.1, 0.5],  # just outside the left edge
    ]
    result = point_in_convex_hull(
        ax, PointInHullInput(points=make_matrix(UNIT_SQUARE), queries=make_matrix(queries))
    )
    assert result.error.code == ""
    assert list(result.inside.values) == [True, False, True, False]


def test_collinear_hull_points_are_degenerate():
    ax = _TestContext()
    result = point_in_convex_hull(
        ax, PointInHullInput(points=make_matrix(COLLINEAR_2D), queries=make_matrix([[1.0, 0.0]]))
    )
    assert result.error.code == "DEGENERATE_INPUT"
