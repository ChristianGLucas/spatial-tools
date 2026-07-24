# spatial-tools

Deterministic computational-spatial algorithms on N-dimensional point sets for
AI agents — built for the [Axiom](https://axiomide.com) marketplace
(`christiangeorgelucas/spatial-tools`).

Wraps [SciPy](https://scipy.org)'s `scipy.spatial` and `scipy.spatial.distance`
(BSD-3-Clause), with NumPy (BSD-3-Clause), to expose nearest-neighbor search,
Delaunay triangulation, Voronoi diagrams, convex hulls, and distance
computations as stateless, deterministic single-input/single-output nodes.

## Use it from your agent or app

Every node in this package is a **live, auto-scaling API endpoint** on the
[Axiom](https://axiomide.com) marketplace — call it from an AI agent or your own
code, with nothing to self-host.

**📦 See it on the marketplace:**
https://dev.axiomide.com/marketplace/christiangeorgelucas/spatial-tools@0.1.1

**Hook it up to an AI agent (MCP).** Add Axiom's hosted MCP server to any MCP
client and every node becomes a typed tool your agent can call — search the
catalog, inspect a schema, and invoke it directly.

```bash
# Claude Code
claude mcp add --transport http axiom https://api.axiomide.com/mcp \
  --header "Authorization: Bearer $AXIOM_API_KEY"
```

Claude Desktop, Cursor, or any config-based client:

```json
{
  "mcpServers": {
    "axiom": {
      "type": "http",
      "url": "https://api.axiomide.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_AXIOM_API_KEY" }
    }
  }
}
```

**Call it from the CLI.**

```bash
axiom invoke christiangeorgelucas/spatial-tools/DelaunayTriangulate --input '{ ... }'
```

**Call it over HTTP.**

```bash
curl -X POST https://api.axiomide.com/invocations/v1/nodes/christiangeorgelucas/spatial-tools/0.1.1/DelaunayTriangulate \
  -H "Authorization: Bearer $AXIOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{ ... }'
```

> Input/output schema for each node is on the marketplace page above, or via
> `axiom inspect node christiangeorgelucas/spatial-tools/DelaunayTriangulate`.

### Get started free

Install the CLI:

```bash
# macOS / Linux — Homebrew
brew install axiomide/tap/axiom

# macOS / Linux — install script
curl -fsSL https://raw.githubusercontent.com/AxiomIDE/axiom-releases/main/install.sh | sh
```

**Windows:** download the `windows/amd64` `.zip` from the
[releases page](https://github.com/AxiomIDE/axiom-releases/releases), unzip it,
and put `axiom.exe` on your `PATH`.

Then `axiom version` to verify, `axiom login` (GitHub or Google) to authenticate,
and create an API key under **Console → API Keys**. Docs and sign-up at
**[axiomide.com](https://axiomide.com)**.

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
