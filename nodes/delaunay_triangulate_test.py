from gen.messages_pb2 import DelaunayInput
from nodes.delaunay_triangulate import delaunay_triangulate
from nodes._test_helpers import COLLINEAR_2D, UNIT_SQUARE, UNIT_CUBE, _TestContext, make_matrix


def test_unit_square_gives_two_triangles():
    ax = _TestContext()
    result = delaunay_triangulate(ax, DelaunayInput(points=make_matrix(UNIT_SQUARE)))
    assert result.error.code == ""
    # A convex quadrilateral triangulates into exactly 2 triangles.
    assert len(result.simplices.rows) == 2
    for row in result.simplices.rows:
        assert len(row.values) == 3  # D+1 = 3 vertex indices per triangle
    # Every point index used must be a valid index into the 4 input points.
    used = {v for row in result.simplices.rows for v in row.values}
    assert used <= {0, 1, 2, 3}
    # Neighbor structure: 2 triangles sharing the square's diagonal share
    # exactly one neighbor each; the other 2 facets of each triangle are on
    # the hull boundary (-1).
    for row in result.neighbors.rows:
        assert len(row.values) == 3
        assert sum(1 for v in row.values if v == -1) == 2


def test_unit_cube_triangulates_3d():
    ax = _TestContext()
    result = delaunay_triangulate(ax, DelaunayInput(points=make_matrix(UNIT_CUBE)))
    assert result.error.code == ""
    assert len(result.simplices.rows) > 0
    for row in result.simplices.rows:
        assert len(row.values) == 4  # D+1 = 4 vertex indices per tetrahedron


def test_collinear_points_are_degenerate():
    ax = _TestContext()
    result = delaunay_triangulate(ax, DelaunayInput(points=make_matrix(COLLINEAR_2D)))
    assert result.error.code == "DEGENERATE_INPUT"


def test_too_few_points_errors():
    ax = _TestContext()
    result = delaunay_triangulate(ax, DelaunayInput(points=make_matrix([[0.0, 0.0], [1.0, 0.0]])))
    assert result.error.code == "INSUFFICIENT_POINTS"


def test_wrong_dimension_errors():
    ax = _TestContext()
    result = delaunay_triangulate(
        ax, DelaunayInput(points=make_matrix([[0.0], [1.0], [2.0], [3.0]]))
    )
    assert result.error.code == "INVALID_ARGUMENT"
