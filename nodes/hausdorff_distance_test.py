from gen.messages_pb2 import HausdorffInput
from nodes.hausdorff_distance import hausdorff_distance
from nodes._test_helpers import UNIT_SQUARE, _TestContext, make_matrix


def test_known_asymmetric_hausdorff():
    ax = _TestContext()
    # A = unit square corners. B = a single point far away at (10, 10).
    # directed(A,B) = max over a in A of dist(a, B) = dist((0,0),(10,10))
    #   = sqrt(200) ~= 14.142 (farthest corner from B is the origin).
    # directed(B,A) = min over a in A of dist(B, a) = dist((10,10),(1,1))
    #   = sqrt(162) ~= 12.728.
    # symmetric = max(...) = directed(A,B) = sqrt(200), achieved from A.
    b = [[10.0, 10.0]]
    result = hausdorff_distance(
        ax, HausdorffInput(points_a=make_matrix(UNIT_SQUARE), points_b=make_matrix(b))
    )
    assert result.error.code == ""
    assert abs(result.distance - 200.0 ** 0.5) < 1e-6
    assert result.from_a is True
    assert list(result.index_pair.values) == [0, 0]  # A's (0,0) is farthest; only 1 point in B


def test_identical_sets_have_zero_hausdorff():
    ax = _TestContext()
    result = hausdorff_distance(
        ax, HausdorffInput(points_a=make_matrix(UNIT_SQUARE), points_b=make_matrix(UNIT_SQUARE))
    )
    assert result.error.code == ""
    assert abs(result.distance) < 1e-9


def test_dimension_mismatch_errors():
    ax = _TestContext()
    result = hausdorff_distance(
        ax,
        HausdorffInput(
            points_a=make_matrix(UNIT_SQUARE), points_b=make_matrix([[0.0, 0.0, 0.0]])
        ),
    )
    assert result.error.code == "DIMENSION_MISMATCH"
