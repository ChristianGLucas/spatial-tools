from gen.messages_pb2 import CdistInput
from nodes.cross_distance_matrix import cross_distance_matrix
from nodes._test_helpers import _TestContext, make_matrix


def test_known_cross_distances():
    ax = _TestContext()
    a = [[0.0, 0.0], [1.0, 0.0]]
    b = [[0.0, 1.0], [3.0, 4.0]]
    result = cross_distance_matrix(ax, CdistInput(points_a=make_matrix(a), points_b=make_matrix(b)))
    assert result.error.code == ""
    assert len(result.distances.rows) == 2
    assert len(result.distances.rows[0].values) == 2
    # (0,0)-(0,1) = 1.0 ; (0,0)-(3,4) = 5.0 (3-4-5 triangle)
    assert abs(result.distances.rows[0].values[0] - 1.0) < 1e-9
    assert abs(result.distances.rows[0].values[1] - 5.0) < 1e-9
    # (1,0)-(0,1) = sqrt(2) ; (1,0)-(3,4) = sqrt(4+16) = sqrt(20)
    assert abs(result.distances.rows[1].values[0] - 2.0 ** 0.5) < 1e-9
    assert abs(result.distances.rows[1].values[1] - 20.0 ** 0.5) < 1e-9


def test_dimension_mismatch_errors():
    ax = _TestContext()
    result = cross_distance_matrix(
        ax, CdistInput(points_a=make_matrix([[0.0, 0.0]]), points_b=make_matrix([[0.0, 0.0, 0.0]]))
    )
    assert result.error.code == "DIMENSION_MISMATCH"
