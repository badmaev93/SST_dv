import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator

H, W = 1650, 3600
lat_inc = np.linspace(-90, 90, H)
lon_arr = np.linspace(0, 360, W)
field_filled = np.random.rand(H, W)
pts = np.random.rand(256 * 256, 2) * [180, 360] - [90, 0]

# Benchmark initialization
t0 = time.time()
for _ in range(100):
    interp = RegularGridInterpolator(
        (lat_inc, lon_arr), field_filled,
        method="linear", bounds_error=False, fill_value=np.nan)
t1 = time.time()
print(f"Init time: {(t1 - t0) / 100 * 1000:.2f} ms")

# Benchmark call
t2 = time.time()
for _ in range(100):
    vals = interp(pts)
t3 = time.time()
print(f"Call time: {(t3 - t2) / 100 * 1000:.2f} ms")
