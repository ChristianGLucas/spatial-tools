from gen.messages_pb2 import ProcrustesInput
from nodes.procrustes_disparity import procrustes_disparity
from nodes._test_helpers import _TestContext, make_matrix


def test_exact_rotation_has_near_zero_disparity():
    ax = _TestContext()
    # points_b is points_a rotated 90 degrees about the origin -- an exact
    # rigid rotation of the same shape, so Procrustes disparity (after
    # scipy's own centering/scaling/rotation alignment) must be ~0.
    a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    b = [[0.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]
    result = procrustes_disparity(ax, ProcrustesInput(points_a=make_matrix(a), points_b=make_matrix(b)))
    assert result.error.code == ""
    assert result.disparity < 1e-6
    assert len(result.aligned_a.rows) == 3
    assert len(result.aligned_b.rows) == 3


def test_dissimilar_shapes_have_positive_disparity():
    ax = _TestContext()
    a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    b = [[0.0, 0.0], [5.0, 0.0], [1.0, 8.0]]  # a stretched/skewed shape, not a's rotation
    result = procrustes_disparity(ax, ProcrustesInput(points_a=make_matrix(a), points_b=make_matrix(b)))
    assert result.error.code == ""
    assert result.disparity > 0.01


def test_shape_mismatch_errors():
    ax = _TestContext()
    a = [[0.0, 0.0], [1.0, 0.0]]
    b = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    result = procrustes_disparity(ax, ProcrustesInput(points_a=make_matrix(a), points_b=make_matrix(b)))
    assert result.error.code == "DIMENSION_MISMATCH"


def test_zero_variance_input_is_degenerate():
    ax = _TestContext()
    a = [[1.0, 1.0], [1.0, 1.0]]  # both rows identical -> no shape
    b = [[0.0, 0.0], [1.0, 1.0]]
    result = procrustes_disparity(ax, ProcrustesInput(points_a=make_matrix(a), points_b=make_matrix(b)))
    assert result.error.code == "DEGENERATE_INPUT"
