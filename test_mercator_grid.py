import numpy as np
import time
from pyproj import Transformer
import functools

_MERC_EXTENT = (-2e7, -2e7, 2e7, 2e7)

def _get_mercator_grid_old(W: int, H: int):
    x_min, y_min, x_max, y_max = _MERC_EXTENT
    xs = np.linspace(x_min, x_max, W)
    ys = np.linspace(y_max, y_min, H)
    X, Y = np.meshgrid(xs, ys)
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lons_grid, lats_grid = inv.transform(X, Y)
    return lons_grid % 360.0, lats_grid

def _get_mercator_grid_new(W: int, H: int):
    x_min, y_min, x_max, y_max = _MERC_EXTENT
    xs = np.linspace(x_min, x_max, W)
    ys = np.linspace(y_max, y_min, H)
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    # ⚡ Bolt: Transform 1D arrays first, then meshgrid to avoid O(N^2) transform overhead
    lons_1d, _ = inv.transform(xs, np.zeros_like(xs))
    _, lats_1d = inv.transform(np.zeros_like(ys), ys)
    lons_grid, lats_grid = np.meshgrid(lons_1d, lats_1d)
    return lons_grid % 360.0, lats_grid

t0 = time.time()
old_lons, old_lats = _get_mercator_grid_old(3600, 1650)
t1 = time.time()
new_lons, new_lats = _get_mercator_grid_new(3600, 1650)
t2 = time.time()

print(f"Old time: {(t1 - t0) * 1000:.2f} ms")
print(f"New time: {(t2 - t1) * 1000:.2f} ms")
print(np.allclose(old_lons, new_lons))
print(np.allclose(old_lats, new_lats))
