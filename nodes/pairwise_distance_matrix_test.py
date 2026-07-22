from gen.messages_pb2 import PdistInput
from nodes.pairwise_distance_matrix import pairwise_distance_matrix
from nodes._test_helpers import UNIT_SQUARE, _TestContext, make_matrix


def test_unit_square_condensed_distances():
    ax = _TestContext()
    result = pairwise_distance_matrix(ax, PdistInput(points=make_matrix(UNIT_SQUARE)))
    assert result.error.code == ""
    assert result.n == 4
    # UNIT_SQUARE = [(0,0),(1,0),(0,1),(1,1)]; pdist's condensed order is
    # (0,1),(0,2),(0,3),(1,2),(1,3),(2,3):
    #   (0,0)-(1,0)=1, (0,0)-(0,1)=1, (0,0)-(1,1)=sqrt(2),
    #   (1,0)-(0,1)=sqrt(2), (1,0)-(1,1)=1, (0,1)-(1,1)=1
    expected = [1.0, 1.0, 2.0 ** 0.5, 2.0 ** 0.5, 1.0, 1.0]
    got = list(result.condensed.values)
    assert len(got) == 6
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-9


def test_cityblock_metric():
    ax = _TestContext()
    result = pairwise_distance_matrix(
        ax, PdistInput(points=make_matrix(UNIT_SQUARE), metric="cityblock")
    )
    assert result.error.code == ""
    # cityblock (0,0)-(1,1) = |1|+|1| = 2, unlike euclidean's sqrt(2).
    assert abs(result.condensed.values[2] - 2.0) < 1e-9


def test_too_few_points_errors():
    ax = _TestContext()
    result = pairwise_distance_matrix(ax, PdistInput(points=make_matrix([[0.0, 0.0]])))
    assert result.error.code == "EMPTY_INPUT"


def test_unknown_metric_errors():
    ax = _TestContext()
    result = pairwise_distance_matrix(
        ax, PdistInput(points=make_matrix(UNIT_SQUARE), metric="not-a-metric")
    )
    assert result.error.code == "INVALID_ARGUMENT"
