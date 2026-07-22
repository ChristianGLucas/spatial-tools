from gen.messages_pb2 import BoundingBoxInput
from nodes.bounding_box import bounding_box
from nodes._test_helpers import UNIT_SQUARE, _TestContext, make_matrix


def test_unit_square_bounding_box():
    ax = _TestContext()
    result = bounding_box(ax, BoundingBoxInput(points=make_matrix(UNIT_SQUARE)))
    assert result.error.code == ""
    assert list(result.min.values) == [0.0, 0.0]
    assert list(result.max.values) == [1.0, 1.0]


def test_asymmetric_points():
    ax = _TestContext()
    points = [[-3.0, 10.0], [5.0, -2.0], [0.0, 0.0]]
    result = bounding_box(ax, BoundingBoxInput(points=make_matrix(points)))
    assert result.error.code == ""
    assert list(result.min.values) == [-3.0, -2.0]
    assert list(result.max.values) == [5.0, 10.0]


def test_empty_input_errors():
    ax = _TestContext()
    result = bounding_box(ax, BoundingBoxInput(points=make_matrix([])))
    assert result.error.code == "EMPTY_INPUT"
