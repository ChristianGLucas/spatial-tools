from gen.messages_pb2 import KNNInput
from nodes.k_nearest_neighbors import k_nearest_neighbors
from nodes._test_helpers import LINE_1D, _TestContext, make_matrix


def test_finds_obvious_nearest_neighbor():
    ax = _TestContext()
    result = k_nearest_neighbors(
        ax, KNNInput(points=make_matrix(LINE_1D), queries=make_matrix([[11.0]]), k=1)
    )
    assert result.error.code == ""
    assert list(result.indices.rows[0].values) == [1]  # LINE_1D[1] == 10.0, closest to 11
    assert abs(result.distances.rows[0].values[0] - 1.0) < 1e-9


def test_k_two_orders_by_distance():
    ax = _TestContext()
    result = k_nearest_neighbors(
        ax, KNNInput(points=make_matrix(LINE_1D), queries=make_matrix([[11.0]]), k=2)
    )
    assert result.error.code == ""
    # nearest is index 1 (10.0, dist 1); 2nd nearest is index 2 (20.0, dist 9)
    assert list(result.indices.rows[0].values) == [1, 2]
    assert abs(result.distances.rows[0].values[0] - 1.0) < 1e-9
    assert abs(result.distances.rows[0].values[1] - 9.0) < 1e-9


def test_cityblock_metric_matches_euclidean_on_1d():
    ax = _TestContext()
    result = k_nearest_neighbors(
        ax,
        KNNInput(points=make_matrix(LINE_1D), queries=make_matrix([[11.0]]), k=1, metric="cityblock"),
    )
    assert result.error.code == ""
    assert abs(result.distances.rows[0].values[0] - 1.0) < 1e-9


def test_k_too_large_errors():
    ax = _TestContext()
    result = k_nearest_neighbors(
        ax, KNNInput(points=make_matrix(LINE_1D), queries=make_matrix([[11.0]]), k=99)
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_dimension_mismatch_errors():
    ax = _TestContext()
    result = k_nearest_neighbors(
        ax, KNNInput(points=make_matrix(LINE_1D), queries=make_matrix([[1.0, 2.0]]), k=1)
    )
    assert result.error.code == "DIMENSION_MISMATCH"


def test_unknown_metric_errors():
    ax = _TestContext()
    result = k_nearest_neighbors(
        ax, KNNInput(points=make_matrix(LINE_1D), queries=make_matrix([[11.0]]), k=1, metric="cosine")
    )
    assert result.error.code == "INVALID_ARGUMENT"
