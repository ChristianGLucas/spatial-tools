from gen.messages_pb2 import ConvexHullInput
from nodes.convex_hull import convex_hull
from nodes._test_helpers import COLLINEAR_2D, UNIT_CUBE, UNIT_SQUARE, _TestContext, make_matrix


def test_unit_square_area_and_perimeter():
    ax = _TestContext()
    result = convex_hull(ax, ConvexHullInput(points=make_matrix(UNIT_SQUARE)))
    assert result.error.code == ""
    assert abs(result.volume - 1.0) < 1e-9  # area, for D=2
    assert abs(result.area - 4.0) < 1e-9  # perimeter, for D=2
    assert set(result.vertices.values) == {0, 1, 2, 3}  # all 4 corners are extreme points


def test_point_inside_square_is_not_a_hull_vertex():
    ax = _TestContext()
    points = UNIT_SQUARE + [[0.5, 0.5]]  # interior point, index 4
    result = convex_hull(ax, ConvexHullInput(points=make_matrix(points)))
    assert result.error.code == ""
    assert 4 not in set(result.vertices.values)
    assert abs(result.volume - 1.0) < 1e-9  # interior point doesn't change the hull's area


def test_unit_cube_volume_and_surface_area():
    ax = _TestContext()
    result = convex_hull(ax, ConvexHullInput(points=make_matrix(UNIT_CUBE)))
    assert result.error.code == ""
    assert abs(result.volume - 1.0) < 1e-9
    assert abs(result.area - 6.0) < 1e-9


def test_collinear_points_are_degenerate():
    ax = _TestContext()
    result = convex_hull(ax, ConvexHullInput(points=make_matrix(COLLINEAR_2D)))
    assert result.error.code == "DEGENERATE_INPUT"


def test_too_few_points_errors():
    ax = _TestContext()
    result = convex_hull(ax, ConvexHullInput(points=make_matrix([[0.0, 0.0], [1.0, 0.0]])))
    assert result.error.code == "INSUFFICIENT_POINTS"
