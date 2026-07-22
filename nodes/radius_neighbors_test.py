from gen.messages_pb2 import RadiusNeighborsInput
from nodes.radius_neighbors import radius_neighbors
from nodes._test_helpers import PLUS_SIGN, _TestContext, make_matrix


def test_finds_all_points_within_radius():
    ax = _TestContext()
    # Query the center of the plus-sign: exactly the 4 arms are at distance
    # 1.0, and the center itself (index 0) is at distance 0 <= radius 1.5.
    result = radius_neighbors(
        ax, RadiusNeighborsInput(points=make_matrix(PLUS_SIGN), queries=make_matrix([[0.0, 0.0]]), radius=1.5)
    )
    assert result.error.code == ""
    assert sorted(result.indices.rows[0].values) == [0, 1, 2, 3, 4]


def test_tight_radius_excludes_diagonal_neighbors():
    ax = _TestContext()
    # radius=1.0 from the center includes only the 4 arms at exactly
    # distance 1.0, plus itself at distance 0 -- NOT the sqrt(2)-away
    # diagonal gaps (there are none here, but this also proves a radius
    # smaller than 1 excludes everything but the query point itself).
    result = radius_neighbors(
        ax, RadiusNeighborsInput(points=make_matrix(PLUS_SIGN), queries=make_matrix([[0.0, 0.0]]), radius=0.5)
    )
    assert result.error.code == ""
    assert list(result.indices.rows[0].values) == [0]


def test_invalid_radius_errors():
    ax = _TestContext()
    result = radius_neighbors(
        ax, RadiusNeighborsInput(points=make_matrix(PLUS_SIGN), queries=make_matrix([[0.0, 0.0]]), radius=0.0)
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_dimension_mismatch_errors():
    ax = _TestContext()
    result = radius_neighbors(
        ax,
        RadiusNeighborsInput(
            points=make_matrix(PLUS_SIGN), queries=make_matrix([[0.0, 0.0, 0.0]]), radius=1.0
        ),
    )
    assert result.error.code == "DIMENSION_MISMATCH"
