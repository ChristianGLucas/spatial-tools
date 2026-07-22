from gen.messages_pb2 import VoronoiInput
from nodes.voronoi_diagram import voronoi_diagram
from nodes._test_helpers import COLLINEAR_2D, UNIT_SQUARE, _TestContext, make_matrix


def test_unit_square_has_one_finite_vertex():
    ax = _TestContext()
    result = voronoi_diagram(ax, VoronoiInput(points=make_matrix(UNIT_SQUARE)))
    assert result.error.code == ""
    # The Voronoi diagram of a square's 4 corners has exactly one finite
    # vertex: the square's center (0.5, 0.5), equidistant from all 4 sites.
    assert len(result.vertices.rows) == 1
    vx, vy = result.vertices.rows[0].values
    assert abs(vx - 0.5) < 1e-9
    assert abs(vy - 0.5) < 1e-9
    # scipy's `regions` list includes one placeholder empty region (its own
    # "point at infinity" convention) alongside the 4 real per-point
    # regions, so `point_region` is the correct way to look up point i's
    # region rather than assuming regions[i]. Each of the 4 input points'
    # own region is unbounded (contains the single finite vertex index 0
    # plus a -1 marking the unbounded side).
    assert len(result.point_region.values) == 4
    for pr in result.point_region.values:
        assert 0 <= pr < len(result.regions.rows)
        region = list(result.regions.rows[pr].values)
        assert sorted(region) == [-1, 0]


def test_collinear_points_are_degenerate():
    ax = _TestContext()
    result = voronoi_diagram(ax, VoronoiInput(points=make_matrix(COLLINEAR_2D)))
    assert result.error.code == "DEGENERATE_INPUT"


def test_too_few_points_errors():
    ax = _TestContext()
    result = voronoi_diagram(ax, VoronoiInput(points=make_matrix([[0.0, 0.0], [1.0, 0.0]])))
    assert result.error.code == "INSUFFICIENT_POINTS"
