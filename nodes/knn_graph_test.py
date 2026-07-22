from gen.messages_pb2 import KNNGraphInput
from nodes.knn_graph import knn_graph
from nodes._test_helpers import PLUS_SIGN, _TestContext, make_matrix


def test_center_points_to_all_four_arms_at_k4():
    ax = _TestContext()
    result = knn_graph(ax, KNNGraphInput(points=make_matrix(PLUS_SIGN), k=4))
    assert result.error.code == ""
    # 5 points * k=4 = 20 directed edges total, none of them self-edges.
    assert len(result.edges.rows) == 20
    for row in result.edges.rows:
        assert row.values[0] != row.values[1]
    # The center (index 0)'s 4 nearest OTHER points are exactly the 4 arms,
    # each at distance 1.0.
    center_edges = [row for row in result.edges.rows if row.values[0] == 0]
    assert sorted(int(row.values[1]) for row in center_edges) == [1, 2, 3, 4]


def test_k1_gives_single_nearest_neighbor_per_point():
    ax = _TestContext()
    result = knn_graph(ax, KNNGraphInput(points=make_matrix(PLUS_SIGN), k=1))
    assert result.error.code == ""
    assert len(result.edges.rows) == 5
    # Point 0 (center)'s single nearest OTHER point is one of the 4 arms
    # (all tied at distance 1.0) -- must be one of them, and the recorded
    # weight must equal that distance.
    idx = next(i for i, row in enumerate(result.edges.rows) if row.values[0] == 0)
    assert int(result.edges.rows[idx].values[1]) in (1, 2, 3, 4)
    assert abs(result.weights.values[idx] - 1.0) < 1e-9


def test_k_too_large_errors():
    ax = _TestContext()
    result = knn_graph(ax, KNNGraphInput(points=make_matrix(PLUS_SIGN), k=99))
    assert result.error.code == "INVALID_ARGUMENT"
