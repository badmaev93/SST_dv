import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

H, W = 1650, 3600
lat_inc = np.linspace(-90, 90, H)
lon_arr = np.linspace(0, 360, W)
field_filled = np.random.rand(H, W)
pts = np.random.rand(256 * 256, 2) * [180, 360] - [90, 0]

t2 = time.time()
for _ in range(100):
    interp = RegularGridInterpolator(
        (lat_inc, lon_arr), field_filled,
        method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts)
t3 = time.time()
print(f"RGI Call time: {(t3 - t2) / 100 * 1000:.2f} ms")

# For map_coordinates, we need to map pts to indices
t4 = time.time()
for _ in range(100):
    # pts[:, 0] maps to lat_inc (-90 to 90) -> (pts[:, 0] + 90) / 180 * (H - 1)
    # pts[:, 1] maps to lon_arr (0 to 360) -> pts[:, 1] / 360 * (W - 1)
    idx_y = (pts[:, 0] - lat_inc[0]) / (lat_inc[-1] - lat_inc[0]) * (H - 1)
    idx_x = (pts[:, 1] - lon_arr[0]) / (lon_arr[-1] - lon_arr[0]) * (W - 1)

    vals_map = map_coordinates(field_filled, [idx_y, idx_x], order=1, mode='constant', cval=np.nan)
t5 = time.time()
print(f"map_coordinates Call time: {(t5 - t4) / 100 * 1000:.2f} ms")
