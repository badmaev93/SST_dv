import numpy as np
import time
from pyproj import Transformer

W, H = 1200, 550
xs = np.linspace(-2e7, 2e7, W)
ys = np.linspace(-2e7, 2e7, H)

tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

t0 = time.time()
for _ in range(100):
    X, Y = np.meshgrid(xs, ys)
    lons_grid, lats_grid = tr.transform(X, Y)
t1 = time.time()
print(f"2D Transform time: {(t1 - t0) / 100 * 1000:.2f} ms")

t2 = time.time()
for _ in range(100):
    lons_1d, _ = tr.transform(xs, np.zeros_like(xs))
    _, lats_1d = tr.transform(np.zeros_like(ys), ys)
    lons_grid2, lats_grid2 = np.meshgrid(lons_1d, lats_1d)
t3 = time.time()
print(f"1D Transform time: {(t3 - t2) / 100 * 1000:.2f} ms")

print(np.allclose(lons_grid, lons_grid2))
print(np.allclose(lats_grid, lats_grid2))
