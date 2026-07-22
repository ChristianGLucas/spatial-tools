from gen.messages_pb2 import PointDistanceInput
from nodes.point_distance import point_distance
from nodes._test_helpers import _TestContext, make_matrix, make_vector


def test_euclidean_3_4_5_triangle():
    ax = _TestContext()
    result = point_distance(
        ax, PointDistanceInput(a=make_vector([0.0, 0.0]), b=make_vector([3.0, 4.0]))
    )
    assert result.error.code == ""
    assert abs(result.distance - 5.0) < 1e-9


def test_cityblock_metric():
    ax = _TestContext()
    result = point_distance(
        ax,
        PointDistanceInput(a=make_vector([0.0, 0.0]), b=make_vector([3.0, 4.0]), metric="cityblock"),
    )
    assert result.error.code == ""
    assert abs(result.distance - 7.0) < 1e-9


def test_chebyshev_metric():
    ax = _TestContext()
    result = point_distance(
        ax,
        PointDistanceInput(a=make_vector([0.0, 0.0]), b=make_vector([3.0, 4.0]), metric="chebyshev"),
    )
    assert result.error.code == ""
    assert abs(result.distance - 4.0) < 1e-9  # max(|3|, |4|)


def test_dimension_mismatch_errors():
    ax = _TestContext()
    result = point_distance(
        ax, PointDistanceInput(a=make_vector([0.0, 0.0]), b=make_vector([1.0, 2.0, 3.0]))
    )
    assert result.error.code == "DIMENSION_MISMATCH"


def test_mahalanobis_without_inv_cov_errors():
    ax = _TestContext()
    result = point_distance(
        ax,
        PointDistanceInput(a=make_vector([0.0, 0.0]), b=make_vector([1.0, 1.0]), metric="mahalanobis"),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_mahalanobis_with_identity_inv_cov_matches_euclidean():
    ax = _TestContext()
    identity = make_matrix([[1.0, 0.0], [0.0, 1.0]])
    result = point_distance(
        ax,
        PointDistanceInput(
            a=make_vector([0.0, 0.0]), b=make_vector([3.0, 4.0]), metric="mahalanobis", inv_cov=identity
        ),
    )
    assert result.error.code == ""
    # Mahalanobis distance with the identity inverse-covariance reduces
    # exactly to euclidean distance.
    assert abs(result.distance - 5.0) < 1e-9
