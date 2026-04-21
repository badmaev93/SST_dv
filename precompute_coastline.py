#!/usr/bin/env python3
"""Растеризация Natural Earth 1:10m в маску океана data/ocean_mask.npy (1200*550).

Область 117-246E пересекает антимеридиан; полигоны клипируются в двух половинах
(117-180E и 180-246E) отдельно, чтобы избежать самопересечений при % 360.
Запускать один раз: python precompute_coastline.py
"""
import os, json, urllib.request
import numpy as np
from matplotlib.path import Path
from pyproj import Transformer
from shapely.geometry import shape, box, Polygon, MultiPolygon
from shapely.validation import make_valid

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GEOJSON_URL   = ("https://raw.githubusercontent.com/nvkelso/"
                 "natural-earth-vector/master/geojson/ne_10m_land.geojson")
GEOJSON_CACHE = os.path.join(DATA_DIR, "ne_10m_land.geojson")

WIDTH  = 1200
HEIGHT = 550

_EARTH_CIRC = 40075016.685578488   # длина экватора WGS-84, м

LON_MIN, LON_MAX = 117.0, 246.0
LAT_MIN, LAT_MAX = 26.0,  72.0

# клип-боксы в стандартных координатах -180..180 с перекрытием 0.5 у антимеридиана
_MARGIN = 0.5
_BOX_W  = box(LON_MIN - _MARGIN, LAT_MIN - _MARGIN,
              180.0 + _MARGIN,    LAT_MAX + _MARGIN)  # 117-180E
_BOX_E  = box(-180.0 - _MARGIN,  LAT_MIN - _MARGIN,
              -(360.0 - LON_MAX) - _MARGIN,
              LAT_MAX + _MARGIN)  # -180..-114E (= 180-246E)


def _mercator_extent():
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, _ = tr.transform(LON_MIN, LAT_MIN)
    x_max    = x_min + (LON_MAX - LON_MIN) / 360.0 * _EARTH_CIRC
    _, y_max = tr.transform(0.0, LAT_MAX)
    _, y_min = tr.transform(0.0, LAT_MIN)
    return x_min, y_min, x_max, y_max


def _to_pixel(lon_std, lat, east_half, tr, x_min, x_max, y_min, y_max):
    """lon_std в -180..180; east_half=True - сдвиг на EARTH_CIRC"""
    try:
        x, y = tr.transform(lon_std, lat)
        if east_half:
            x += _EARTH_CIRC
        px = (x - x_min) / (x_max - x_min) * WIDTH
        py = (y_max - y)  / (y_max - y_min) * HEIGHT
        return px, py
    except Exception:
        return None


def _rasterize_polygon(poly, east_half, land, tr, x_min, x_max, y_min, y_max):
    """Ray-cast одного полигона shapely в массив land"""
    if poly.is_empty:
        return

    # Exterior ring
    pts = []
    for lon, lat in poly.exterior.coords:
        r = _to_pixel(lon, lat, east_half, tr, x_min, x_max, y_min, y_max)
        if r is not None:
            pts.append(r)
    if len(pts) < 3:
        return

    arr  = np.array(pts, dtype=np.float64)
    bx0  = max(0,      int(np.floor(arr[:, 0].min())))
    bx1  = min(WIDTH,  int(np.ceil (arr[:, 0].max())) + 1)
    by0  = max(0,      int(np.floor(arr[:, 1].min())))
    by1  = min(HEIGHT, int(np.ceil (arr[:, 1].max())) + 1)
    if bx0 >= bx1 or by0 >= by1:
        return

    cols = np.arange(bx0, bx1) + 0.5
    rows = np.arange(by0, by1) + 0.5
    C, R = np.meshgrid(cols, rows)
    pts_rc  = np.column_stack([C.ravel(), R.ravel()])
    inside  = Path(arr, closed=True).contains_points(pts_rc)
    land[by0:by1, bx0:bx1] |= inside.reshape(by1 - by0, bx1 - bx0)

    # дыры - стираем
    for hole in poly.interiors:
        hpts = []
        for lon, lat in hole.coords:
            r = _to_pixel(lon, lat, east_half, tr, x_min, x_max, y_min, y_max)
            if r is not None:
                hpts.append(r)
        if len(hpts) < 3:
            continue
        harr  = np.array(hpts, dtype=np.float64)
        hbx0  = max(0,      int(np.floor(harr[:, 0].min())))
        hbx1  = min(WIDTH,  int(np.ceil (harr[:, 0].max())) + 1)
        hby0  = max(0,      int(np.floor(harr[:, 1].min())))
        hby1  = min(HEIGHT, int(np.ceil (harr[:, 1].max())) + 1)
        if hbx0 >= hbx1 or hby0 >= hby1:
            continue
        hcols  = np.arange(hbx0, hbx1) + 0.5
        hrows  = np.arange(hby0, hby1) + 0.5
        HC, HR = np.meshgrid(hcols, hrows)
        h_pts_rc = np.column_stack([HC.ravel(), HR.ravel()])
        hinside  = Path(harr, closed=True).contains_points(h_pts_rc)
        land[hby0:hby1, hbx0:hbx1] &= ~hinside.reshape(hby1 - hby0, hbx1 - hbx0)


def _rasterize_geom(clipped, east_half, land, tr, x_min, x_max, y_min, y_max):
    if clipped is None or clipped.is_empty:
        return
    gt = clipped.geom_type
    if gt == "Polygon":
        _rasterize_polygon(clipped, east_half, land, tr, x_min, x_max, y_min, y_max)
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for g in clipped.geoms:
            if g.geom_type == "Polygon":
                _rasterize_polygon(g, east_half, land, tr, x_min, x_max, y_min, y_max)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. скачивание GeoJSON
    if not os.path.exists(GEOJSON_CACHE):
        print("Downloading Natural Earth 1:10m land polygons …")
        urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_CACHE)
        print(f"  saved {os.path.getsize(GEOJSON_CACHE)//1024} KB")
    else:
        print(f"Using cached ({os.path.getsize(GEOJSON_CACHE)//1024} KB)")

    with open(GEOJSON_CACHE, encoding="utf-8") as f:
        geo = json.load(f)
    print(f"Features in GeoJSON: {len(geo['features'])}")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min, x_max, y_max = _mercator_extent()
    print(f"Mercator extent: X [{x_min:.0f}, {x_max:.0f}]  Y [{y_min:.0f}, {y_max:.0f}]")
    print(f"East clip box: {_BOX_E.bounds}")

    # 2. растеризация
    land = np.zeros((HEIGHT, WIDTH), dtype=bool)

    for i, feat in enumerate(geo["features"]):
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)

        # западная половина 117-180E
        w = geom.intersection(_BOX_W)
        _rasterize_geom(w, False, land, tr, x_min, x_max, y_min, y_max)

        # восточная половина 180-246E
        e = geom.intersection(_BOX_E)
        _rasterize_geom(e, True, land, tr, x_min, x_max, y_min, y_max)

        if i % 5 == 0:
            print(f"  feature {i}: w_empty={w.is_empty}, e_empty={e.is_empty}")

    ocean_mask = ~land
    out_path   = os.path.join(DATA_DIR, "ocean_mask.npy")
    np.save(out_path, ocean_mask)
    print(f"\nSaved: {out_path}")
    print(f"  ocean={ocean_mask.sum():,}  land={land.sum():,}  total={WIDTH*HEIGHT:,}")
