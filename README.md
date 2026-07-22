# spatial-tools

Deterministic computational-spatial algorithms on N-dimensional point sets for
AI agents — built for the [Axiom](https://axiom.dev) marketplace
(`christiangeorgelucas/spatial-tools`).

Wraps [SciPy](https://scipy.org)'s `scipy.spatial` and `scipy.spatial.distance`
(BSD-3-Clause), with NumPy (BSD-3-Clause), to expose nearest-neighbor search,
Delaunay triangulation, Voronoi diagrams, convex hulls, and distance
computations as stateless, deterministic single-input/single-output nodes.

## Nodes

| Node | What it does |
|---|---|
| `DelaunayTriangulate` | Delaunay triangulation of a 2D/3D point set (simplices + neighbor structure) |
| `VoronoiDiagram` | Voronoi diagram of a 2D/3D point set (vertices, regions, ridges) |
| `ConvexHull` | Convex hull of a general N-dimensional point set (vertices, simplices, volume, area) |
| `KNearestNeighbors` | KD-tree-accelerated k-nearest-neighbor query |
| `RadiusNeighbors` | KD-tree-accelerated ball (radius) query |
| `QueryPairs` | All pairs of points within a distance, inside one point set |
| `NearestSinglePoint` | Single nearest neighbor to one query point |
| `KNNGraph` | Spatial k-nearest-neighbor graph (edges + weights) |
| `PairwiseDistanceMatrix` | Condensed pairwise distance vector (pdist) across the full scipy metric family |
| `CrossDistanceMatrix` | Cross distance matrix between two point sets (cdist) |
| `PointDistance` | Distance between two specific points under a chosen metric |
| `HausdorffDistance` | Symmetric Hausdorff distance between two point sets |
| `ProcrustesDisparity` | Procrustes shape-alignment disparity between two point sets |
| `PointInConvexHull` | Point-in-convex-hull membership test |
| `FindSimplex` | Locate the Delaunay simplex containing a query point + barycentric weights |
| `BoundingBox` | Axis-aligned bounding box of a point set |

Distinct from `geometry-tools` (2D planar polygon/line geometry via JTS),
`mesh-tools` (3D triangle-mesh operations via trimesh), and `sklearn-tools`
(fits ML algorithms on data) — this package owns the `scipy.spatial`
algorithms those don't cover: Delaunay/Voronoi tessellation, KD-tree-native
queries, and the full scipy distance-metric surface, on raw N-dimensional
point sets.

Every node is stateless and fully deterministic (no randomness, no
wall-clock, no network) and hard-caps its input point count/dimensionality
to stay under Axiom's ~4 MiB transport limit — see each node's description
in `axiom.yaml` for the exact bound.

## License

MIT — see [LICENSE](./LICENSE). Wraps SciPy and NumPy (both BSD-3-Clause).
