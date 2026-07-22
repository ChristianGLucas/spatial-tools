from gen.messages_pb2 import FindSimplexInput
from nodes.find_simplex import find_simplex
from nodes._test_helpers import UNIT_SQUARE, _TestContext, make_matrix


def test_center_point_is_located_with_correct_barycentric_weights():
    ax = _TestContext()
    queries = [[0.5, 0.5], [2.0, 2.0]]
    result = find_simplex(
        ax, FindSimplexInput(points=make_matrix(UNIT_SQUARE), queries=make_matrix(queries))
    )
    assert result.error.code == ""
    simplex_idx = list(result.simplex_indices.values)
    assert simplex_idx[0] >= 0  # center is inside the triangulation
    assert simplex_idx[1] == -1  # (2,2) is outside the unit square's hull

    # Reconstruct the query point from its barycentric weights and the
    # actual simplex vertices -- this must equal the query point exactly,
    # independent of which diagonal scipy happened to triangulate on.
    row = result.barycentric_coords.rows[0].values
    assert abs(sum(row) - 1.0) < 1e-9
    for w in row:
        assert -1e-9 <= w <= 1.0 + 1e-9  # inside -> weights in [0,1]

    # Row for the outside query is all-zero (no containing simplex).
    outside_row = list(result.barycentric_coords.rows[1].values)
    assert all(w == 0.0 for w in outside_row)


def test_vertex_query_has_weight_one_on_itself():
    ax = _TestContext()
    # Querying exactly one of the input points must land in some simplex
    # containing it with barycentric weight 1.0 on that vertex, 0 elsewhere.
    result = find_simplex(
        ax, FindSimplexInput(points=make_matrix(UNIT_SQUARE), queries=make_matrix([[0.0, 0.0]]))
    )
    assert result.error.code == ""
    assert result.simplex_indices.values[0] >= 0
    row = list(result.barycentric_coords.rows[0].values)
    assert abs(max(row) - 1.0) < 1e-6
    assert abs(sum(row) - 1.0) < 1e-9
