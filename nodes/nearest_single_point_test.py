from gen.messages_pb2 import NearestSinglePointInput
from nodes.nearest_single_point import nearest_single_point
from nodes._test_helpers import LINE_1D, _TestContext, make_matrix, make_vector


def test_finds_obvious_nearest_neighbor():
    ax = _TestContext()
    result = nearest_single_point(
        ax, NearestSinglePointInput(points=make_matrix(LINE_1D), query=make_vector([11.0]))
    )
    assert result.error.code == ""
    assert result.index == 1  # LINE_1D[1] == 10.0, closest to 11
    assert abs(result.distance - 1.0) < 1e-9


def test_exact_match_has_zero_distance():
    ax = _TestContext()
    result = nearest_single_point(
        ax, NearestSinglePointInput(points=make_matrix(LINE_1D), query=make_vector([20.0]))
    )
    assert result.error.code == ""
    assert result.index == 2
    assert result.distance == 0.0


def test_dimension_mismatch_errors():
    ax = _TestContext()
    result = nearest_single_point(
        ax, NearestSinglePointInput(points=make_matrix(LINE_1D), query=make_vector([1.0, 2.0]))
    )
    assert result.error.code == "DIMENSION_MISMATCH"


def test_empty_query_errors():
    ax = _TestContext()
    result = nearest_single_point(
        ax, NearestSinglePointInput(points=make_matrix(LINE_1D), query=make_vector([]))
    )
    assert result.error.code == "EMPTY_INPUT"
