import numpy as np
from scipy.spatial import cKDTree

from gen.axiom_context import AxiomContext
from gen.messages_pb2 import KNNGraphInput, KNNGraphResult
from nodes._common import (
    NodeInputError,
    array_to_int_matrix,
    array_to_vector,
    check_matrix_shape,
    err,
    matrix_to_array,
)


def knn_graph(ax: AxiomContext, input: KNNGraphInput) -> KNNGraphResult:
    """Build a spatial k-nearest-neighbor GRAPH over `points` (euclidean):
    for every point, find its k nearest OTHER points (self excluded) via
    scipy.spatial.cKDTree, and return the resulting directed edge list
    (i -> j meaning "j is one of i's k nearest neighbors") with per-edge
    distances. Not symmetrized — union/intersect the edges yourself for a
    mutual/undirected graph.
    """
    try:
        check_matrix_shape(input.points, name="points")
        points = matrix_to_array(input.points)
        n = points.shape[0]

        k = input.k
        max_k = n - 1
        if k < 1 or k > max_k:
            raise NodeInputError(
                "INVALID_ARGUMENT", f"k must satisfy 1 <= k <= len(points)-1 = {max_k} (got {k})"
            )

        tree = cKDTree(points)
        # Query k+1 to include self, then drop the self-match per row.
        dist, idx = tree.query(points, k=k + 1, p=2)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)

        edges = []
        weights = []
        for i in range(n):
            count = 0
            for j, dj in zip(idx[i], dist[i]):
                if int(j) == i:
                    continue
                edges.append((i, int(j)))
                weights.append(float(dj))
                count += 1
                if count == k:
                    break

        return KNNGraphResult(
            edges=array_to_int_matrix(np.array(edges, dtype=np.int64) if edges else np.empty((0, 2))),
            weights=array_to_vector(weights),
        )
    except NodeInputError as e:
        return KNNGraphResult(error=e.error)
    except Exception as e:
        return KNNGraphResult(error=err("GRAPH_FAILED", str(e)))
