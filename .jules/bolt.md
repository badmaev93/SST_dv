## 2026-11-06 - PyProj Transformer Meshgrid Bottleneck
**Learning:** In map rendering loops (like `app.py`'s `_render_mercator`), repeatedly running `np.meshgrid` and applying `pyproj.Transformer.from_crs(...).transform(X,Y)` over the entire grid array is a massive bottleneck (~0.07-0.10s per request). Because the extent bounds are constant, the transformation outputs are entirely deterministic for a given `width` and `height`.
**Action:** When seeing `pyproj.Transformer` (or similar heavy transformations) applied over grids where the bounding boxes are static, extract the grid generation logic into a helper function and use `@functools.lru_cache` to memoize the resulting `numpy` coordinate arrays. Ensure to use an LRU cache with a small `maxsize` to prevent unbounded memory growth from large coordinate arrays.

## 2024-05-18 - Colormap Generation Performance
**Learning:** The colormap generation in `_make_lut` used nested Python loops and manual linear interpolation. This is slow and scales poorly. Vectorized interpolation with `np.interp` is up to 100x faster and simplifies the code significantly.
**Action:** When implementing custom linear interpolations across arrays in Python, always look for vectorized numpy alternatives like `np.interp` instead of using Python loops.

## 2024-05-24 - Slippy Map Tiling Overhead

**Learning:** When serving dynamically generated map tiles, functions called per tile that operate on the entire global field (like `_fill_land_nans` using `scipy.ndimage.distance_transform_edt` for the 550x1200 array) introduce severe cumulative overhead, adding seconds to the load time for a set of 50 map tiles.

**Action:** Extract expensive operations that are independent of the tile coordinates out of the per-tile rendering pipeline and memoize them in a context cache function (`_get_render_context`) that is already called per request.

## 2026-11-10 - PyProj Transformation Optimization with Separable Coordinates
**Learning:** The transformation from Web Mercator (EPSG:3857) to WGS84 (EPSG:4326) is strictly orthogonal (x maps directly to longitude, y maps directly to latitude). Using `np.meshgrid` to generate a 2D array and then applying `pyproj.Transformer` on the flattened 2D arrays is extremely inefficient for separable projections, with a cost of O(W*H).
**Action:** When working with strictly orthogonal/separable projections, perform the coordinate transformation on the 1D vectors *before* applying `np.meshgrid`. This reduces the transformation cost from O(W*H) to O(W+H), resulting in ~20x faster coordinate grid generation.
