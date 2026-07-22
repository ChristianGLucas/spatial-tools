from gen.messages_pb2 import QueryPairsInput
from nodes.query_pairs import query_pairs
from nodes._test_helpers import PLUS_SIGN, _TestContext, make_matrix


def test_finds_pairs_within_distance():
    ax = _TestContext()
    # PLUS_SIGN: center (0) is distance 1.0 from each arm (1,2,3,4); arms
    # are distance sqrt(2)~=1.414 or 2.0 apart from each other. distance=1.1
    # should find exactly the 4 (center, arm) pairs.
    result = query_pairs(ax, QueryPairsInput(points=make_matrix(PLUS_SIGN), distance=1.1))
    assert result.error.code == ""
    pairs = {tuple(sorted((int(row.values[0]), int(row.values[1])))) for row in result.pairs.rows}
    assert pairs == {(0, 1), (0, 2), (0, 3), (0, 4)}
    for row in result.pairs.rows:
        assert row.values[0] < row.values[1]  # i < j convention


def test_wider_distance_includes_adjacent_arms():
    ax = _TestContext()
    # sqrt(2) ~= 1.41421356 is the distance between adjacent arms (e.g.
    # (1,0) to (0,1)); distance=1.5 must include those pairs too.
    result = query_pairs(ax, QueryPairsInput(points=make_matrix(PLUS_SIGN), distance=1.5))
    assert result.error.code == ""
    pairs = {tuple(sorted((int(row.values[0]), int(row.values[1])))) for row in result.pairs.rows}
    assert (1, 3) in pairs  # (1,0)-(0,1): sqrt(2)
    assert len(pairs) > 4


def test_invalid_distance_errors():
    ax = _TestContext()
    result = query_pairs(ax, QueryPairsInput(points=make_matrix(PLUS_SIGN), distance=-1.0))
    assert result.error.code == "INVALID_ARGUMENT"
