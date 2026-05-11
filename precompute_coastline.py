#!/usr/bin/env python3
"""Растеризация Natural Earth 1:10m в маску океана data/ocean_mask.npy.

Запускать один раз: python precompute_coastline.py
"""
import os, json, urllib.request
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import shape, box
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GEOJSON_URL   = ("https://raw.githubusercontent.com/nvkelso/"
                 "natural-earth-vector/master/geojson/ne_10m_land.geojson")
GEOJSON_CACHE = os.path.join(DATA_DIR, "ne_10m_land.geojson")

WIDTH        = 1200
HEIGHT       = 550
WIDTH_HIRES  = 3600
HEIGHT_HIRES = 1650

_EARTH_CIRC = 40075016.685578488

LON_MIN, LON_MAX = 117.0, 246.0
LAT_MIN, LAT_MAX =  26.0,  72.0

_BOX_FILTER = box(LON_MIN - 1.0, LAT_MIN - 1.0, LON_MAX + 1.0, LAT_MAX + 1.0)


def _mercator_extent():
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, _ = tr.transform(LON_MIN, LAT_MIN)
    x_max    = x_min + (LON_MAX - LON_MIN) / 360.0 * _EARTH_CIRC
    _, y_max = tr.transform(0.0, LAT_MAX)
    _, y_min = tr.transform(0.0, LAT_MIN)
    return x_min, y_min, x_max, y_max


def _normalize_ring(coords):
    """Нормализовать lon < 0 -> lon + 360 в пределах одного кольца.
    """
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])
    if np.median(lons) < 0:
        lons += 360.0
    return list(zip(lons, lats))


def _ring_in_region(coords):
    """Быстрая bbox-проверка"""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (max(lons) >= LON_MIN - 1 and min(lons) <= LON_MAX + 1 and
            max(lats) >= LAT_MIN - 1 and min(lats) <= LAT_MAX + 1)


def _ring_to_pixels(coords, tr, x_min, x_max, y_min, y_max, W, H):
    """Перевести кольцо в пиксельные координаты"""
    pts = []
    for lon, lat in coords:
        try:
            if lon > 180.0:
                x, y = tr.transform(lon - 360.0, lat)
                x += _EARTH_CIRC
            else:
                x, y = tr.transform(lon, lat)
            px = (x - x_min) / (x_max - x_min) * W
            py = (y_max - y) / (y_max - y_min) * H
            pts.append((round(px), round(py)))
        except Exception:
            pass
    return pts


def build_land_mask(features, W, H, tr, x_min, x_max, y_min, y_max):
    """Растеризовать список GeoJSON features в land-маску (H, W) bool

    Два прохода:
      1. Рис. внешние кольца на ext_img — залив. сушу
      2. Рис. внутренние кольца на hole_img - вычит.

    """
    ext_img   = Image.new("L", (W, H), 0)
    ext_draw  = ImageDraw.Draw(ext_img)
    hole_img  = Image.new("L", (W, H), 0)
    hole_draw = ImageDraw.Draw(hole_img)

    for feat in features:
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)

        gt = geom.geom_type
        if gt == "Polygon":
            sub_polys = [geom]
        elif gt in ("MultiPolygon", "GeometryCollection"):
            sub_polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        else:
            continue

        for poly in sub_polys:
            if poly.is_empty:
                continue

            ext_coords = _normalize_ring(list(poly.exterior.coords))

            if not _ring_in_region(ext_coords):
                continue

            pts = _ring_to_pixels(ext_coords, tr, x_min, x_max, y_min, y_max, W, H)
            if len(pts) >= 3:
                ext_draw.polygon(pts, fill=255)

            for ring in poly.interiors:
                hcoords = _normalize_ring(list(ring.coords))
                hpts = _ring_to_pixels(hcoords, tr, x_min, x_max, y_min, y_max, W, H)
                if len(hpts) >= 3:
                    hole_draw.polygon(hpts, fill=255)

    land  = np.asarray(ext_img,  dtype=bool)
    holes = np.asarray(hole_img, dtype=bool)
    return land & ~holes


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(GEOJSON_CACHE):
        print("Скачивние 1:10m полигоны суши …")
        urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_CACHE)
        print(f"  сохранено {os.path.getsize(GEOJSON_CACHE)//1024} KB")
    else:
        print(f"Использ. кэш. GeoJSON ({os.path.getsize(GEOJSON_CACHE)//1024} KB)")

    with open(GEOJSON_CACHE, encoding="utf-8") as f:
        geo = json.load(f)
    features = geo["features"]
    print(f"Features в GeoJSON: {len(features)}")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min, x_max, y_max = _mercator_extent()
    print(f"Экстент Меркатора: X [{x_min:.0f}, {x_max:.0f}]  Y [{y_min:.0f}, {y_max:.0f}]")

    print(f"\nРасттеризация {WIDTH}×{HEIGHT} …")
    land = build_land_mask(features, WIDTH, HEIGHT, tr, x_min, x_max, y_min, y_max)
    ocean_mask = ~land
    out_path = os.path.join(DATA_DIR, "ocean_mask.npy")
    np.save(out_path, ocean_mask)
    print(f"Сохранено: {out_path}  ocean={ocean_mask.sum():,}  land={land.sum():,}")

    print(f"\nРастеризация {WIDTH_HIRES}×{HEIGHT_HIRES} (hires) …")
    land_hi = build_land_mask(features, WIDTH_HIRES, HEIGHT_HIRES,
                              tr, x_min, x_max, y_min, y_max)
    ocean_hires = ~land_hi
    out_hi = os.path.join(DATA_DIR, "ocean_mask_hires.npy")
    np.save(out_hi, ocean_hires)
    print(f"Saved: {out_hi}  ocean={ocean_hires.sum():,}  land={land_hi.sum():,}")
