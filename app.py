"""
SST Dashboard v5 — FastAPI backend
"""
import os, io, math, logging, time
from typing import Optional

# метка старта - сброс кэша клиентов при рестарте
_SERVER_START_TS = str(int(time.time()))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
from scipy.ndimage import zoom as nd_zoom
from PIL import Image
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="SST Dashboard v5")

# ленивая загрузка из .npz
_cache: dict = {}

def _load(name: str) -> dict:
    if name not in _cache:
        path = os.path.join(DATA_DIR, f"{name}.npz")
        if not os.path.exists(path):
            raise HTTPException(500, f"Data not precomputed: {name}.npz — run precompute.py first.")
        _cache[name] = dict(np.load(path, allow_pickle=True))
    return _cache[name]

# LUT цветошкал (256*3, uint8)
def _make_lut(name: str) -> np.ndarray:
    n   = 256
    lut = np.zeros((n, 3), dtype=np.uint8)
    t   = np.linspace(0, 1, n)

    def c(v): return int(np.clip(v, 0, 255))

    if name == "RdYlBu_r":
        colors = [
            (0.000, (49,  54,  149)),
            (0.125, (69,  117, 180)),
            (0.250, (116, 173, 209)),
            (0.375, (171, 217, 233)),
            (0.500, (255, 255, 191)),
            (0.625, (254, 224, 144)),
            (0.750, (253, 174, 97)),
            (0.875, (244, 109, 67)),
            (1.000, (165, 0,   38)),
        ]
        for i, v in enumerate(t):
            for k in range(len(colors) - 1):
                v0, c0 = colors[k]
                v1, c1 = colors[k + 1]
                if v0 <= v <= v1:
                    f = (v - v0) / (v1 - v0)
                    lut[i] = [c(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                    break
    elif name == "RdBu_r":
        colors = [
            (0.000, (5,   48,  97)),
            (0.250, (33,  102, 172)),
            (0.375, (103, 169, 207)),
            (0.450, (209, 229, 240)),
            (0.500, (247, 247, 247)),
            (0.550, (253, 219, 199)),
            (0.625, (239, 138, 98)),
            (0.750, (178, 24,  43)),
            (1.000, (103, 0,   31)),
        ]
        for i, v in enumerate(t):
            for k in range(len(colors) - 1):
                v0, c0 = colors[k]
                v1, c1 = colors[k + 1]
                if v0 <= v <= v1:
                    f = (v - v0) / (v1 - v0)
                    lut[i] = [c(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                    break
    elif name == "Viridis":
        colors = [
            (0.000, (68,  1,   84)),
            (0.250, (59,  82,  139)),
            (0.500, (33,  145, 140)),
            (0.750, (94,  201, 98)),
            (1.000, (253, 231, 37)),
        ]
        for i, v in enumerate(t):
            for k in range(len(colors) - 1):
                v0, c0 = colors[k]
                v1, c1 = colors[k + 1]
                if v0 <= v <= v1:
                    f = (v - v0) / (v1 - v0)
                    lut[i] = [c(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                    break
    elif name == "YlOrRd":
        colors = [
            (0.000, (255, 255, 204)),
            (0.250, (254, 217, 118)),
            (0.500, (253, 141, 60)),
            (0.750, (227, 26,  28)),
            (1.000, (128, 0,   38)),
        ]
        for i, v in enumerate(t):
            for k in range(len(colors) - 1):
                v0, c0 = colors[k]
                v1, c1 = colors[k + 1]
                if v0 <= v <= v1:
                    f = (v - v0) / (v1 - v0)
                    lut[i] = [c(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                    break
    elif name == "Hot_r":
        # Hot_r: бел-жёлт-красн-чёрн
        colors = [
            (0.000, (255, 255, 255)),
            (0.333, (255, 255, 0)),
            (0.667, (255, 0,   0)),
            (1.000, (0,   0,   0)),
        ]
        for i, v in enumerate(t):
            for k in range(len(colors) - 1):
                v0, c0 = colors[k]
                v1, c1 = colors[k + 1]
                if v0 <= v <= v1:
                    f = (v - v0) / (v1 - v0)
                    lut[i] = [c(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                    break
    else:
        lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(n, dtype=np.uint8)
    return lut


COLORMAPS = {
    "s1": ("RdYlBu_r", False),
    "s2": ("RdBu_r",   True),
    "s3": ("Viridis",  False),
    "s4": ("YlOrRd",   False),
    "s5": ("RdBu_r",   True),
    "s6": ("Hot_r",    False),
    "s7": ("RdBu_r",   True),
}

_LUTS = {nm: _make_lut(nm) for nm in set(n for n, _ in COLORMAPS.values())}

# метаданные секций
SECTIONS = [
    {"id": "s1", "label": "Климатологическое среднее ТПО",          "unit": "°C",              "symmetric": False},
    {"id": "s2", "label": "Линейный тренд ТПО (p<0.05)",            "unit": "°C/дек",          "symmetric": True},
    {"id": "s3", "label": "Коэффициент детерминации тренда (R²)",   "unit": "",                "symmetric": False},
    {"id": "s4", "label": "Дисперсия ТПО",                          "unit": "°C²",             "symmetric": False},
    {"id": "s5", "label": "Тренд дисперсии ТПО",                    "unit": "°C²/дек",         "symmetric": True},
    {"id": "s7", "label": "Аномалии ТПО",                           "unit": "°C",              "symmetric": True},
]

ESS_LABELS = ["Дек–Фев", "Мар–Май", "Июн–Авг", "Сен–Ноя"]
MON_LABELS = ["Январь","Февраль","Март","Апрель","Май","Июнь",
              "Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]

# группы месяцев: ДЯФ=0, МАМ=1, ИИА=2, СОЯ=3
_SEASONAL_MONTHS = [[11, 0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10]]

# границы перцентилей для цветошкалы
def _pct_bounds(percentile: float) -> tuple[float, float]:
    """Вернуть (lo, hi) перцентили для вычисления vmin/vmax"""
    p = float(percentile)
    if p >= 100:
        return (0.0, 100.0)
    elif p >= 99.9:
        return (0.1, 99.9)
    else:
        return (1.0, 99.0)

# рендер поля в PNG-байты
def _render(field: np.ndarray, cmap: str, land_mask: np.ndarray,
            vmin_ov=None, vmax_ov=None, symmetric: bool = False,
            percentile: float = 99) -> tuple[bytes, float, float]:
    lut   = _LUTS[cmap]
    valid = field[~land_mask & np.isfinite(field)]

    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    elif vmin_ov is not None:
        vmin, vmax = float(vmin_ov), float(vmax_ov)
    elif symmetric:
        lo, hi = _pct_bounds(percentile)
        if hi >= 100:
            vm = float(np.max(np.abs(valid)))
        else:
            vm = float(np.percentile(np.abs(valid), hi))
        vmin, vmax = -vm, vm
    else:
        lo, hi = _pct_bounds(percentile)
        if lo <= 0 and hi >= 100:
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
        else:
            vmin = float(np.percentile(valid, lo))
            vmax = float(np.percentile(valid, hi))

    if vmax == vmin:
        vmax = vmin + 1.0

    NLAT, NLON = field.shape
    img  = np.zeros((NLAT, NLON, 4), dtype=np.uint8)
    norm = np.where(np.isfinite(field),
                    np.clip((field - vmin) / (vmax - vmin), 0, 1), 0.0)
    idx  = (norm * 255).astype(np.uint8)

    ocean = ~land_mask & np.isfinite(field)
    img[ocean, :3] = lut[idx[ocean]]
    img[ocean, 3]  = 255

    # NaN-океан - полупрозрачный тёмный
    img[~land_mask & ~np.isfinite(field)] = [20, 20, 30, 80]

    pil = Image.fromarray(img, "RGBA")
    # 2x увеличение для сглаживания
    ow, oh = pil.size
    pil = pil.resize((ow * 2, oh * 2), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), vmin, vmax


def _compute_s7_field(season_type: str, season_idx: int,
                      year: int, clim_period: str) -> np.ndarray:
    meta      = _load("meta")
    years_a   = meta["years"]
    months_a  = meta["months"]
    ess_a     = meta["ess_idx"]
    land_mask = meta["land_mask"]
    raw_sst   = _load("sst_raw")["sst"]
    s7        = _load("s7")
    clim_key  = f"clim_{clim_period}"
    if clim_key not in s7:
        raise HTTPException(400, f"Unknown clim_period: {clim_period}")
    clim = s7[clim_key]

    if season_type == "monthly":
        idx = np.where((years_a == year) & (months_a - 1 == season_idx))[0]
        if len(idx) == 0:
            return np.full(land_mask.shape, np.nan, dtype=np.float32)
        field = raw_sst[idx[0]].astype(np.float32) - clim[season_idx]
    else:
        # ESS / seasonal: average the months in this season for this year
        months_in_season = _SEASONAL_MONTHS[season_idx % 4]
        frames, clim_frames = [], []
        for mo in months_in_season:
            mo1 = (mo % 12) + 1
            yr  = year if mo < 12 else year - 1
            ix  = np.where((years_a == yr) & (months_a == mo1))[0]
            if len(ix):
                frames.append(raw_sst[ix[0]].astype(np.float32))
                clim_frames.append(clim[mo % 12].astype(np.float32))
        if not frames:
            return np.full(land_mask.shape, np.nan, dtype=np.float32)
        field = np.nanmean(frames, axis=0) - np.nanmean(clim_frames, axis=0)

    field[land_mask] = np.nan
    return field.astype(np.float32)


# кэш S5 по месяцам
_s5_monthly_cache: dict = {}

def _get_s5_monthly_field(month_idx: int, meta: dict) -> np.ndarray:
    """Тренд межгодовой дисперсии по месяцу (скол. окно 10 лет)"""
    if month_idx in _s5_monthly_cache:
        return _s5_monthly_cache[month_idx]

    raw       = _load("sst_raw")["sst"]
    years_a   = meta["years"]
    months_a  = meta["months"]
    land_mask = meta["land_mask"]
    NLAT, NLON = land_mask.shape

    # один кадр на год для месяца
    all_years = sorted(set(int(y) for y in years_a))
    frames, yr_list = [], []
    for yr in all_years:
        ix = np.where((years_a == yr) & (months_a - 1 == month_idx))[0]
        if len(ix):
            frames.append(raw[ix[0]].astype(np.float32))
            yr_list.append(float(yr))

    if len(frames) < 10:
        result = np.full((NLAT, NLON), np.nan, dtype=np.float32)
        _s5_monthly_cache[month_idx] = result
        return result

    stack  = np.stack(frames)            # (N, NLAT, NLON)
    yr_arr = np.array(yr_list)
    N      = len(yr_list)
    W      = 10                          # 10-year rolling window

    # скользящая дисперсия
    rv_list, ry_list = [], []
    for start in range(N - W + 1):
        rv_list.append(np.nanvar(stack[start:start + W], axis=0))   # (NLAT, NLON)
        ry_list.append(yr_arr[start + W // 2])

    rv    = np.stack(rv_list)            # (n_win, NLAT, NLON)
    ry    = np.array(ry_list, dtype=np.float64)
    ry_c  = ry - ry.mean()
    sx2   = float(np.dot(ry_c, ry_c))

    if sx2 == 0:
        result = np.full((NLAT, NLON), np.nan, dtype=np.float32)
        _s5_monthly_cache[month_idx] = result
        return result

    n_win  = len(ry)
    rv_2d  = rv.reshape(n_win, NLAT * NLON)
    valid  = np.sum(np.isfinite(rv_2d), axis=0)                     # (NLAT*NLON,)
    filled = np.where(np.isfinite(rv_2d), rv_2d, 0.0)

    # OLS вект. (x10 per decade)
    slopes = (np.dot(ry_c, filled) / sx2).reshape(NLAT, NLON).astype(np.float32)
    slopes[valid.reshape(NLAT, NLON) < 5] = np.nan
    slopes[land_mask] = np.nan
    result = (slopes * 10.0).astype(np.float32)

    _s5_monthly_cache[month_idx] = result
    logger.info("s5_monthly[%d] computed: finite=%d", month_idx, int(np.sum(np.isfinite(result))))
    return result


# константы Меркатора (дол. 117-246, шир. 26-72)
_EARTH_CIRC = 40075016.685578488  # meters, WGS-84 equatorial circumference

def _mercator_extent():
    """Return (x_min, y_min, x_max, y_max) in EPSG:3857 for the data domain."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = tr.transform(117.0, 26.0)
    x_max         = x_min + (246.0 - 117.0) / 360.0 * _EARTH_CIRC
    _, y_max      = tr.transform(0.0, 72.0)
    return x_min, y_min, x_max, y_max

_MERC_EXTENT = _mercator_extent()   # computed once at import time

# маска океана - hi-res (3600*1650) с BICUBIC даунскейлом; фолбэк 1200*550
_ocean_mask_hires: Optional[np.ndarray] = None  # (1650, 3600) bool
_ocean_mask:       Optional[np.ndarray] = None  # (550, 1200) bool

_MASK_HIRES_W, _MASK_HIRES_H = 3600, 1650
_MASK_STD_W,   _MASK_STD_H   = 1200, 550

def _load_ocean_mask_hires() -> Optional[np.ndarray]:
    global _ocean_mask_hires
    if _ocean_mask_hires is not None:
        return _ocean_mask_hires
    path = os.path.join(DATA_DIR, "ocean_mask_hires.npy")
    if not os.path.exists(path):
        return None
    m = np.load(path)
    if m.shape != (_MASK_HIRES_H, _MASK_HIRES_W):
        logger.warning("ocean_mask_hires shape %s unexpected", m.shape)
        return None
    _ocean_mask_hires = m.astype(bool)
    logger.info("Hi-res ocean mask loaded: %d×%d ocean=%d",
                _MASK_HIRES_W, _MASK_HIRES_H, int(_ocean_mask_hires.sum()))
    return _ocean_mask_hires

def _load_ocean_mask(width: int = 1200, height: int = 550) -> Optional[np.ndarray]:
    """Return ocean mask True=ocean as float32 (0–1), scaled to (height, width).

    Uses hi-res source (3600×1650) with BICUBIC downsampling for anti-aliased
    coastline clipping.  Falls back to 1200×550 binary mask if hires is absent.
    """
    from PIL import Image as _PIL_m
    hires = _load_ocean_mask_hires()
    if hires is not None:
        if width == _MASK_HIRES_W and height == _MASK_HIRES_H:
            return hires.astype(np.float32)
        img = _PIL_m.fromarray(hires.astype(np.uint8) * 255, "L")
        img = img.resize((width, height), resample=_PIL_m.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr

    # Fallback: low-res binary mask
    global _ocean_mask
    if _ocean_mask is not None:
        if width == _MASK_STD_W and height == _MASK_STD_H:
            return _ocean_mask.astype(np.float32)
    path = os.path.join(DATA_DIR, "ocean_mask.npy")
    if not os.path.exists(path):
        logger.warning("ocean_mask.npy not found")
        return None
    mask = np.load(path)
    if mask.shape != (_MASK_STD_H, _MASK_STD_W):
        logger.warning("ocean_mask.npy shape %s != (%d,%d)", mask.shape, _MASK_STD_H, _MASK_STD_W)
        return None
    _ocean_mask = mask.astype(bool)
    logger.info("Std ocean mask loaded: %d ocean pixels", int(_ocean_mask.sum()))
    if width == _MASK_STD_W and height == _MASK_STD_H:
        return _ocean_mask.astype(np.float32)
    img = _PIL_m.fromarray(_ocean_mask.astype(np.uint8) * 255, "L")
    img = img.resize((width, height), resample=_PIL_m.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def _fill_land_nans(field: np.ndarray, land_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Заполнить NaN суши ближайшими значениями океана для сглаживания береговой линии"""
    from scipy.ndimage import distance_transform_edt
    nan_mask = ~np.isfinite(field)
    if not nan_mask.any():
        return field

    # пиксели для заполнения
    if land_mask is not None:
        fill_target = nan_mask & land_mask     # only land NaN pixels
    else:
        fill_target = nan_mask                 # all NaN (fallback)

    # граничные строки/столбцы не заполняем - цветовые артефакты при проекции Меркатора
    fill_target[[0, -1], :] = False
    fill_target[:, [0, -1]] = False

    if not fill_target.any():
        return field                           # nothing to fill

    # фон EDT = конечные пиксели океана
    bg = np.isfinite(field) & (True if land_mask is None else ~land_mask)
    if not bg.any():
        bg = np.isfinite(field)               # fallback: any finite pixel
    if not bg.any():
        filled = field.copy()
        filled[fill_target] = 0.0
        return filled

    _, indices = distance_transform_edt(~bg, return_distances=True, return_indices=True)
    r = np.clip(indices[0], 0, field.shape[0] - 1)
    c = np.clip(indices[1], 0, field.shape[1] - 1)
    filled = field.copy()
    filled[fill_target] = field[r, c][fill_target]
    return filled


def _overlay_contours(img, vals: np.ndarray, finite: np.ndarray,
                      vmin: float, vmax: float, ocean_mask_s,
                      show_labels: bool = False,
                      lut: "np.ndarray | None" = None) -> "PIL.Image.Image":
    """Наложить изолинии на PIL RGBA изображение (2x рендер через Agg, даунскейл LANCZOS)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image as _PIL

        H, W = vals.shape
        n_levels = 10
        levels = np.linspace(vmin, vmax, n_levels + 2)[1:-1]

        # сглаживаем поле чтобы изолинии следовали крупным структурам ТПО
        from scipy.ndimage import gaussian_filter as _gf
        _filled  = np.where(finite, vals, 0.0)
        _weights = _gf(_filled,                   sigma=4.0)
        _wsum    = _gf(finite.astype(np.float32), sigma=4.0)
        _smooth  = np.where(_wsum > 0.05, _weights / np.maximum(_wsum, 1e-6), np.nan)
        vals_c   = np.where(finite, _smooth, np.nan)

        # рендер 2x, даунскейл LANCZOS = бесплатный AA
        SCALE = 2
        DPI   = 150   # physical DPI of the render pass
        fw    = W * SCALE / DPI
        fh    = H * SCALE / DPI

        fig = Figure(figsize=(fw, fh), dpi=DPI, facecolor="none")
        ax  = fig.add_axes([0, 0, 1, 1], facecolor="none")
        ax.patch.set_alpha(0.0)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)   # inverted Y → row 0 at top of buffer
        ax.axis("off")

        CONT_COLOR = (20/255, 30/255, 60/255, 55/255)
        CS = ax.contour(vals_c, levels=levels,
                        colors=[CONT_COLOR], linewidths=1.0)

        if show_labels:
            try:
                fmt_comma = lambda x: f"{x:.1f}".replace(".", ",")
                # цвет подписей: белый на тёмном, тёмный на светлом - по яркости LUT
                if lut is not None:
                    label_colors = []
                    for lv in levels:
                        norm_v = (lv - vmin) / max(vmax - vmin, 1e-10)
                        li = int(np.clip(norm_v * 255, 0, 255))
                        r, g, b = lut[li] / 255.0
                        lum = 0.299 * r + 0.587 * g + 0.114 * b
                        label_colors.append(
                            (1.0, 1.0, 1.0, 0.92) if lum < 0.5
                            else (0.08, 0.10, 0.25, 0.92)
                        )
                else:
                    label_colors = [(0.08, 0.12, 0.24, 0.9)]
                ax.clabel(CS, inline=True, fontsize=6 * SCALE, fmt=fmt_comma,
                          colors=label_colors)
            except Exception as le:
                logger.debug("clabel failed: %s", le)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # buffer_rgba() - сырые RGBA байты; facecolor='none' -> фон alpha=0
        raw  = canvas.buffer_rgba()
        ren  = canvas.get_renderer()
        W2   = int(ren.width)
        H2   = int(ren.height)
        buf_arr = np.frombuffer(raw, dtype=np.uint8).reshape(H2, W2, 4).copy()
        fig.clf(); del fig

        # даунскейл до нужного разрешения через LANCZOS
        cont_hi      = _PIL.fromarray(buf_arr, "RGBA")
        cont_overlay = cont_hi.resize((W, H), _PIL.LANCZOS)

        # маска изолиний над сушей
        if ocean_mask_s is not None:
            cont_arr = np.asarray(cont_overlay).copy()
            cont_arr[~ocean_mask_s, 3] = 0
            cont_overlay = _PIL.fromarray(cont_arr, "RGBA")

        return _PIL.alpha_composite(img, cont_overlay)

    except Exception as e:
        logger.warning("Contour overlay failed: %s", e)
        return img

    except Exception as e:
        logger.warning("Contour overlay failed: %s", e)
        return img


def _render_mercator(field: np.ndarray, lat_arr: np.ndarray, lon_arr: np.ndarray,
                     cmap: str, vmin: float, vmax: float,
                     land_mask: Optional[np.ndarray] = None,
                     width: int = 1200, height: int = 550,
                     scale: int = 1,
                     blur: bool = True,
                     contours: bool = True,
                     contour_labels: bool = False) -> bytes:
    """Рендер 2D поля ТПО в Mercator-проекцию (RGBA PNG).

    field    : 2D (nlat * nlon), NaN = суша или нет данных (p>=0.05 для тренда)
    lat_arr  : 1-D, может убывать
    lon_arr  : 1-D, 0-360
    land_mask: 2D bool, True = суша ERA5
    scale    : масштаб вывода (1=1200*550, 2=2400*1100 и т.д.)
    """
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    W = width  * scale
    H = height * scale

    x_min, y_min, x_max, y_max = _MERC_EXTENT

    xs = np.linspace(x_min, x_max, W)
    ys = np.linspace(y_max, y_min, H)
    X, Y = np.meshgrid(xs, ys)

    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lons_grid, lats_grid = inv.transform(X, Y)
    lons_grid = lons_grid % 360.0

    # ось lat должна возрастать
    if lat_arr[0] > lat_arr[-1]:
        lat_inc   = lat_arr[::-1]
        field_rc  = field[::-1, :]
        lm_rc     = land_mask[::-1, :] if land_mask is not None else None
    else:
        lat_inc   = lat_arr
        field_rc  = field
        lm_rc     = land_mask

    # заполняем только NaN суши - NaN океана (p>=0.05) остаются прозрачными
    field_filled = _fill_land_nans(field_rc, lm_rc)

    interp = RegularGridInterpolator(
        (lat_inc, lon_arr), field_filled,
        method="linear", bounds_error=False, fill_value=np.nan)

    pts  = np.stack([lats_grid.ravel(), lons_grid.ravel()], axis=1)
    vals = interp(pts).reshape(H, W)

    # мягкое сглаживание края домена
    lat_step     = float(abs(lat_arr[1] - lat_arr[0]))
    lon_step     = float(abs(lon_arr[1] - lon_arr[0]))
    FEATHER_CELLS = 4.0
    dist_to_edge = np.minimum.reduce([
        (lats_grid - float(lat_arr.min())) / lat_step,
        (float(lat_arr.max()) - lats_grid) / lat_step,
        (lons_grid - float(lon_arr.min())) / lon_step,
        (float(lon_arr.max()) - lons_grid) / lon_step,
    ])
    edge_mask = dist_to_edge < 0   # strictly outside domain
    vals[edge_mask] = np.nan

    # прибрежное заполнение в Меркаторе: ERA5 пиксели суши в области Natural Earth океана
    # оставляем дыры значимости (era5_land_merc=False) нетронутыми
    if lm_rc is not None:
        from scipy.ndimage import distance_transform_edt as _edt_c
        lm_nn = RegularGridInterpolator(
            (lat_inc, lon_arr), lm_rc.astype(np.float32),
            method="nearest", bounds_error=False, fill_value=1.0)
        era5_land_merc = lm_nn(pts).reshape(H, W) > 0.5
        coastal_gap = ~np.isfinite(vals) & ~edge_mask & era5_land_merc
        src_ok      = np.isfinite(vals) & ~edge_mask
        if coastal_gap.any() and src_ok.any():
            _, _nn = _edt_c(~src_ok, return_indices=True)
            vals[coastal_gap] = vals[_nn[0][coastal_gap], _nn[1][coastal_gap]]

    # цветизация - NaN пиксели остаются прозрачными
    lut      = _LUTS[cmap]
    finite   = np.isfinite(vals)
    safe     = np.where(finite, vals, vmin)
    norm     = np.clip((safe - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    idx      = (norm * 255).astype(np.uint8)

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = lut[idx]
    rgba[~finite, 3] = 0   # NaN pixels: alpha=0

    from PIL import Image as _PIL, ImageEnhance as _IE, ImageFilter as _IF

    # Гауссово размытие: NaN-пиксели заполняем ближайшим океаном, край остаётся чёрным
    if blur:
        from scipy.ndimage import distance_transform_edt
        nf          = ~finite
        nf_interior = nf & ~edge_mask   # coast + significance holes, not domain boundary
        if nf_interior.any() and finite.any():
            _, nn = distance_transform_edt(~finite | edge_mask, return_indices=True)
            rgba[nf_interior, :3] = rgba[nn[0][nf_interior], nn[1][nf_interior], :3]
        rgb_arr = _PIL.fromarray(rgba[:, :, :3], "RGB")
        rgb_arr = rgb_arr.filter(_IF.GaussianBlur(radius=6.0))
        rgba[:, :, :3] = np.asarray(rgb_arr)

    # маска Natural Earth: обрезаем цветовое поле по береговой линии
    ocean_alpha = _load_ocean_mask(W, H)   # float32 0–1, shape (H, W)
    ocean_mask_s = None                    # binary bool, used for contour masking
    if ocean_alpha is not None:
        # заостряем мягкую маску BICUBIC (0.35-0.65) для чёткого AA обрезания
        ocean_alpha_sharp = np.clip((ocean_alpha - 0.35) / 0.30, 0.0, 1.0)
        ocean_mask_s = ocean_alpha > 0.5
        # Ocean pixels: full opacity (colour clip); land pixels: transparent
        rgba[:, :, 3] = np.clip(
            np.where(finite, 255.0, 0.0) * ocean_alpha_sharp, 0, 255
        ).astype(np.uint8)
    else:
        rgba[finite, 3] = 255

    # мягкий фэдер края (3 ячейки ERA5) против прямоугольного артефакта
    FEATHER_CELLS = 3.0
    feather = np.clip(dist_to_edge / FEATHER_CELLS, 0.0, 1.0)
    rgba[:, :, 3] = np.clip(rgba[:, :, 3].astype(np.float32) * feather, 0, 255).astype(np.uint8)
    rgba[edge_mask, :] = 0

    img = _PIL.fromarray(rgba, "RGBA")
    r, g, b, a = img.split()
    rgb = _PIL.merge("RGB", (r, g, b))
    rgb = _IE.Color(rgb).enhance(1.3)
    r2, g2, b2 = rgb.split()
    img = _PIL.merge("RGBA", (r2, g2, b2, a))

    # изолинии (пропускаем если contours=False)
    if contours:
        img = _overlay_contours(img, vals, finite, vmin, vmax, ocean_mask_s,
                                show_labels=contour_labels, lut=lut)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# PNG-кэш
_png_cache: dict = {}

def _compute_full_field(section: str, season_type: str, season_idx: int,
                        year: Optional[int], clim_period: str,
                        meta: dict) -> tuple[np.ndarray, Optional[float], Optional[float]]:
    """Вычислить 2D поле. Вернуть (field, vmin_override, vmax_override)"""
    land_mask = meta["land_mask"]
    cmap, sym = COLORMAPS[section]

    if section == "s7":
        years_a = meta["years"]
        if year is None:
            year = int(years_a[-1])
        s7 = _load("s7")
        anom_vm = float(s7.get("anom_vm_99", s7["anom_vm"]))

        if season_type in ("ess", "seasonal", "monthly"):
            field = _compute_s7_field(season_type, season_idx, year, clim_period)
        else:
            raw_sst  = _load("sst_raw")["sst"]
            months_a = meta["months"]
            clim     = s7[f"clim_{clim_period}"]
            frames, clim_frames = [], []
            for mo1 in range(1, 13):
                ix = np.where((years_a == year) & (months_a == mo1))[0]
                if len(ix):
                    frames.append(raw_sst[ix[0]].astype(np.float32))
                    clim_frames.append(clim[mo1 - 1].astype(np.float32))
            if not frames:
                field = np.full(land_mask.shape, np.nan, dtype=np.float32)
            else:
                field = np.nanmean(frames, axis=0) - np.nanmean(clim_frames, axis=0)
            field[land_mask] = np.nan
        return field, -anom_vm, anom_vm

    elif season_type == "seasonal":
        months_in = _SEASONAL_MONTHS[season_idx % 4]
        data      = _load(section)
        mon_key   = f"monthly_{clim_period}" if f"monthly_{clim_period}" in data else "monthly"
        fields    = [data[mon_key][m].astype(np.float32) for m in months_in]
        field     = np.nanmean(fields, axis=0)
        field[land_mask] = np.nan
        return field, None, None

    elif season_type == "annual":
        data    = _load(section)
        mon_key = f"monthly_{clim_period}" if f"monthly_{clim_period}" in data else "monthly"
        fields  = [data[mon_key][m].astype(np.float32) for m in range(12)]
        field   = np.nanmean(fields, axis=0)
        field[land_mask] = np.nan
        return field, None, None

    elif section == "s5" and season_type == "monthly":
        # если есть предвычисленные данные по периоду - используем
        data_s5   = _load("s5")
        period_key = f"monthly_{clim_period}"
        if period_key in data_s5:
            field = data_s5[period_key][season_idx].astype(np.float32)
            field[land_mask] = np.nan
        else:
            field = _get_s5_monthly_field(season_idx, meta)
        return field, None, None
    else:
        data       = _load(section)
        key2       = "monthly" if season_type == "monthly" else "ess"
        period_key = f"{key2}_{clim_period}"
        field = data[period_key][season_idx].astype(np.float32) if period_key in data else data[key2][season_idx].astype(np.float32)
        return field, None, None


def _get_field_png(section: str, season_type: str, season_idx: int,
                   year: Optional[int] = None,
                   clim_period: str = "1991_2020",
                   percentile: float = 99,
                   half: Optional[str] = None,
                   scale: int = 1,
                   blur: bool = True,
                   contours: bool = True,
                   contour_labels: bool = False,
                   thumb: bool = False) -> tuple[bytes, float, float]:
    key = (section, season_type, season_idx, year, clim_period, percentile, half, scale, blur,
           contours, contour_labels, thumb)
    if key in _png_cache:
        return _png_cache[key]

    meta      = _load("meta")
    lat_arr   = np.asarray(meta["lat"]).ravel()
    lon_arr   = np.asarray(meta["lon"]).ravel()
    land_mask = meta["land_mask"]
    cmap, sym = COLORMAPS[section]

    field, vmin_ov, vmax_ov = _compute_full_field(
        section, season_type, season_idx, year, clim_period, meta)

    # для s7 выбираем anom_vm по параметру percentile
    if section == "s7" and vmin_ov is not None:
        s7 = _load("s7")
        pct = float(percentile)
        if pct >= 100:
            anom_vm = float(s7.get("anom_vm_100", s7["anom_vm"]))
        elif pct >= 99.9:
            anom_vm = float(s7.get("anom_vm_999", s7["anom_vm"]))
        else:
            anom_vm = float(s7.get("anom_vm_99", s7["anom_vm"]))
        vmin_ov, vmax_ov = -anom_vm, anom_vm

    # vmin/vmax по всему полю
    if vmin_ov is None:
        valid = field[~land_mask & np.isfinite(field)]
        if valid.size == 0:
            vmin_ov, vmax_ov = 0.0, 1.0
        elif sym:
            lo, hi = _pct_bounds(percentile)
            vm = float(np.max(np.abs(valid))) if hi >= 100 else float(np.percentile(np.abs(valid), hi))
            vmin_ov, vmax_ov = -vm, vm
        else:
            lo, hi = _pct_bounds(percentile)
            vmin_ov = float(np.min(valid)) if lo <= 0 else float(np.percentile(valid, lo))
            vmax_ov = float(np.max(valid)) if hi >= 100 else float(np.percentile(valid, hi))
    if vmax_ov == vmin_ov:
        vmax_ov = vmin_ov + 1.0

    # миниатюра: средний рендер с изолиниями, без масштабирования
    out_w = 700 if thumb else 1200
    out_h = 320 if thumb else 550
    png = _render_mercator(field, lat_arr, lon_arr, cmap, float(vmin_ov), float(vmax_ov),
                           land_mask=land_mask,
                           width=out_w, height=out_h,
                           scale=1 if thumb else scale,
                           blur=blur,
                           contours=contours if thumb else contours,
                           contour_labels=False if thumb else contour_labels)
    result = (png, float(vmin_ov), float(vmax_ov))
    _png_cache[key] = result
    return result


# XYZ-тайл кэш и рендерер
_TILE_SIZE   = 256
_tile_cache: dict = {}
# (field, lat_arr, lon_arr, land_mask, cmap, vmin, vmax) на ключ рендера
_field_render_cache: dict = {}
# ленивый transformer pyproj 3857->4326
_MERC2GEO = None

def _get_merc2geo():
    global _MERC2GEO
    if _MERC2GEO is None:
        from pyproj import Transformer
        _MERC2GEO = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return _MERC2GEO


def _get_render_context(section, season_type, season_idx, year, clim_period, percentile):
    """Вернуть (field, lat_arr, lon_arr, land_mask, cmap, vmin, vmax) с кэшированием"""
    key = (section, season_type, season_idx, year, clim_period, percentile)
    if key in _field_render_cache:
        return _field_render_cache[key]

    meta      = _load("meta")
    lat_arr   = np.asarray(meta["lat"]).ravel()
    lon_arr   = np.asarray(meta["lon"]).ravel()
    land_mask = meta["land_mask"]
    cmap, sym = COLORMAPS[section]

    field, vmin_ov, vmax_ov = _compute_full_field(
        section, season_type, season_idx, year, clim_period, meta)

    if section == "s7" and vmin_ov is not None:
        s7 = _load("s7")
        pct = float(percentile)
        anom_vm = float(s7.get("anom_vm_100" if pct >= 100 else
                                "anom_vm_999" if pct >= 99.9 else
                                "anom_vm_99",  s7["anom_vm"]))
        vmin_ov, vmax_ov = -anom_vm, anom_vm

    if vmin_ov is None:
        valid = field[~land_mask & np.isfinite(field)]
        if valid.size == 0:
            vmin_ov, vmax_ov = 0.0, 1.0
        elif sym:
            lo, hi = _pct_bounds(percentile)
            vm = float(np.max(np.abs(valid))) if hi >= 100 else float(np.percentile(np.abs(valid), hi))
            vmin_ov, vmax_ov = -vm, vm
        else:
            lo, hi = _pct_bounds(percentile)
            vmin_ov = float(np.min(valid)) if lo <= 0 else float(np.percentile(valid, lo))
            vmax_ov = float(np.max(valid)) if hi >= 100 else float(np.percentile(valid, hi))
    if vmax_ov == vmin_ov:
        vmax_ov = vmin_ov + 1.0

    result = (field, lat_arr, lon_arr, land_mask, cmap, float(vmin_ov), float(vmax_ov))
    _field_render_cache[key] = result
    return result


def _render_tile_bytes(field, lat_arr, lon_arr, land_mask,
                        cmap, vmin, vmax,
                        tx_min, ty_min, tx_max, ty_max,
                        blur: bool = True) -> bytes:
    """Рендер одного 256*256 WebP-тайла для bounding box в EPSG:3857"""
    from PIL import Image as _PIL, ImageFilter as _IF
    W = H = _TILE_SIZE

    # RegularGridInterpolator требует возрастающей оси lat
    if lat_arr[0] > lat_arr[-1]:
        lat_inc  = lat_arr[::-1]
        field_rc = field[::-1]
        lm_rc    = land_mask[::-1]
    else:
        lat_inc  = lat_arr
        field_rc = field
        lm_rc    = land_mask

    field_filled = _fill_land_nans(field_rc, lm_rc)

    # Меркатор-сетка по центрам пикселей тайла
    tw = tx_max - tx_min
    th = ty_max - ty_min
    xs = np.linspace(tx_min + tw / (2 * W), tx_max - tw / (2 * W), W)
    ys = np.linspace(ty_max - th / (2 * H), ty_min + th / (2 * H), H)
    xs_g, ys_g = np.meshgrid(xs, ys)

    tr = _get_merc2geo()
    lons_f, lats_f = tr.transform(xs_g.ravel(), ys_g.ravel())

    # сдвиг долгот в диапазон ERA5 (антимеридиан)
    lon_min = float(lon_arr.min())
    lon_max = float(lon_arr.max())
    lons_n = np.where(lons_f < lon_min, lons_f + 360.0, lons_f)
    lons_n = np.where(lons_n > lon_max, lons_n - 360.0, lons_n)

    pts = np.stack([lats_f, lons_n], axis=1)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (lat_inc, lon_arr), field_filled,
        method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(H, W)

    # пиксели за пределами домена ERA5 - NaN
    outside = ((lats_f < float(lat_arr.min())) | (lats_f > float(lat_arr.max())) |
               (lons_n < lon_min) | (lons_n > lon_max))
    vals[outside.reshape(H, W)] = np.nan

    lut    = _LUTS[cmap]
    finite = np.isfinite(vals)
    safe   = np.where(finite, vals, vmin)
    norm   = np.clip((safe - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    idx    = (norm * 255).astype(np.uint8)

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = lut[idx]

    # заполняем NaN ближайшим цветом и размываем
    if blur and (~finite).any() and finite.any():
        from scipy.ndimage import distance_transform_edt
        nf = ~finite
        _, nn = distance_transform_edt(nf, return_indices=True)
        rgba[nf, :3] = rgba[nn[0][nf], nn[1][nf], :3]
        rgb_img = _PIL.fromarray(rgba[:, :, :3], "RGB")
        rgb_img = rgb_img.filter(_IF.GaussianBlur(radius=3.0))
        rgba[:, :, :3] = np.asarray(rgb_img)

    rgba[finite, 3] = 255

    # маска океана для области тайла
    ocean_mask_data = _load_ocean_mask(1200, 550)
    if ocean_mask_data is not None:
        dx0, dy0, dx1, dy1 = _MERC_EXTENT
        om_cols = np.clip(((xs_g - dx0) / ((dx1 - dx0) / 1200)).astype(int), 0, 1199)
        om_rows = np.clip(((dy1 - ys_g) / ((dy1 - dy0) / 550)).astype(int), 0, 549)
        tile_ocean = ocean_mask_data[om_rows.ravel(), om_cols.ravel()].reshape(H, W)
        rgba[ tile_ocean, 3]  = np.where(finite[tile_ocean], 255, 0)
        rgba[~tile_ocean, 3]  = 0
        rgba[~tile_ocean, :3] = 0

    buf = io.BytesIO()
    _PIL.fromarray(rgba, "RGBA").save(buf, format="WEBP", lossless=True, quality=80)
    return buf.getvalue()


def _get_tile(section, season_type, season_idx, year, clim_period,
               percentile, blur, z, x, y) -> bytes:
    """Fetch from tile_cache or render a 256×256 WebP tile."""
    key = (section, season_type, season_idx, year, clim_period, percentile, blur, z, x, y)
    if key in _tile_cache:
        return _tile_cache[key]

    HALF = _EARTH_CIRC / 2
    n    = 2 ** z
    tm   = _EARTH_CIRC / n          # metres per tile
    tx_min = x * tm - HALF
    tx_max = tx_min + tm
    ty_max = HALF - y * tm          # XYZ: y=0 is north
    ty_min = ty_max - tm

    # вне домена - прозрачный тайл сразу
    dx0, dy0, dx1, dy1 = _MERC_EXTENT
    if tx_max < dx0 or tx_min > dx1 or ty_max < dy0 or ty_min > dy1:
        from PIL import Image as _PIL
        buf = io.BytesIO()
        _PIL.new("RGBA", (_TILE_SIZE, _TILE_SIZE), (0, 0, 0, 0)).save(
            buf, format="WEBP", lossless=True)
        result = buf.getvalue()
        _tile_cache[key] = result
        return result

    field, lat_arr, lon_arr, land_mask, cmap, vmin, vmax = _get_render_context(
        section, season_type, season_idx, year, clim_period, percentile)

    result = _render_tile_bytes(field, lat_arr, lon_arr, land_mask,
                                 cmap, vmin, vmax,
                                 tx_min, ty_min, tx_max, ty_max, blur)
    _tile_cache[key] = result
    return result


def _get_tick_values(section: str) -> np.ndarray:
    """Вернуть предвычисленные перцентильные тики для легенды"""
    data = _load(section)
    if "pcts" in data:
        return data["pcts"].astype(np.float64)
    # Fallback: 5 evenly spaced ticks
    return np.array([])


def _fmt_tick_prec(v: float, prec: int) -> str:
    """Формат числа с заданным кол-вом знаков, десятичный разделитель запятая"""
    s = f"{v:.{prec}f}"
    return s.replace('.', ',')


# API маршруты

@app.get("/api/meta")
def api_meta():
    try:
        meta = _load("meta")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load meta.npz: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to load meta.npz: {e}")

    try:
        lat_arr = np.asarray(meta["lat"]).ravel()
        lon_arr = np.asarray(meta["lon"]).ravel()
    except Exception as e:
        logger.error("Cannot read lat/lon from meta: %s", e, exc_info=True)
        raise HTTPException(500, f"Cannot read lat/lon: {e}")

    # разбивка по антимеридиану
    try:
        lon_split_idx = int(np.searchsorted(lon_arr, 180.0))
        lon_split_val = float(lon_arr[lon_split_idx]) if lon_split_idx < len(lon_arr) else 180.0
    except Exception as e:
        logger.warning("lon split computation failed: %s", e)
        lon_split_idx = int(len(lon_arr) // 2)
        lon_split_val = 180.0

    # date_strings - обработка отсутствия или bytes dtype
    try:
        if "date_strings" in meta:
            dates = []
            for d in meta["date_strings"]:
                dates.append(d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d))
        else:
            years_a  = np.asarray(meta["years"]).ravel()
            months_a = np.asarray(meta["months"]).ravel()
            dates = [f"{int(y):04d}-{int(m):02d}-01" for y, m in zip(years_a, months_a)]
    except Exception as e:
        logger.warning("date_strings fallback failed: %s", e)
        dates = []

    # годы
    try:
        years = sorted(set(int(y) for y in np.asarray(meta["years"]).ravel()))
    except Exception as e:
        logger.warning("years extraction failed: %s", e)
        years = []

    # последний полный год (все 12 месяцев в данных)
    try:
        from collections import Counter as _Counter
        yr_count = _Counter(int(y) for y in np.asarray(meta["years"]).ravel())
        complete_yrs = sorted(y for y, c in yr_count.items() if c == 12)
        last_complete_year = int(complete_yrs[-1]) if complete_yrs else (years[-1] if years else None)
    except Exception as e:
        logger.warning("last_complete_year failed: %s", e)
        last_complete_year = years[-1] if years else None

    # anom_vm - из meta, затем s7.npz, иначе 2.6 по умолчанию
    anom_vm = 2.6
    try:
        if "anom_vm" in meta:
            anom_vm = float(np.asarray(meta["anom_vm"]).flat[0])
        else:
            s7 = _load("s7")
            for key in ("anom_vm_99", "anom_vm", "anom_vm_999", "anom_vm_100"):
                if key in s7:
                    anom_vm = float(np.asarray(s7[key]).flat[0])
                    break
    except Exception as e:
        logger.warning("anom_vm fallback, using 2.6: %s", e)

    lat = lat_arr.tolist()
    lon = lon_arr.tolist()
    x_min, y_min, x_max, y_max = _MERC_EXTENT
    logger.info("api_meta: lat %s..%s  lon %s..%s  years %s  dates %d",
                lat[0], lat[-1], lon[0], lon[-1],
                f"{years[0]}–{years[-1]}" if years else "?", len(dates))

    # навигационный экстент - чуть шире видимой области
    _NAV_LON_LO, _NAV_LON_HI = 120.0, 245.0
    _NAV_LAT_LO, _NAV_LAT_HI =  26.0,  68.0
    try:
        from pyproj import Transformer as _Tr
        _tr = _Tr.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        _dx_lo, _dy_lo = _tr.transform(_NAV_LON_LO, _NAV_LAT_LO)
        _, _dy_hi      = _tr.transform(0.0, _NAV_LAT_HI)
        # x_min - Меркатор x при lon_arr.min(); восточная граница арифметически
        _dx_hi = x_min + (_NAV_LON_HI - float(lon_arr.min())) / 360.0 * _EARTH_CIRC
        data_extent = [_dx_lo, _dy_lo, _dx_hi, _dy_hi]
    except Exception as e:
        logger.warning("data_extent computation failed: %s", e)
        data_extent = [x_min, y_min, x_max, y_max]

    return {
        "lat":                {"min": float(min(lat)), "max": float(max(lat)), "n": len(lat)},
        "lon":                {"min": float(min(lon)), "max": float(max(lon)), "n": len(lon)},
        "lon_split_idx":      lon_split_idx,
        "lon_split_val":      lon_split_val,
        "dates":              dates,
        "years":              years,
        "last_complete_year": last_complete_year,
        "sections":           SECTIONS,
        "ess_labels":         ESS_LABELS,
        "monthly_labels":     MON_LABELS,
        "anom_vm":            anom_vm,
        "mercator_extent":    [x_min, y_min, x_max, y_max],
        "data_extent":        data_extent,
        "server_ts":          _SERVER_START_TS,
    }


@app.get("/api/land_mask.png")
def api_land_mask_png():
    """Статичная маска суши: непрозрачный (#f0efea) над сушей, прозрачный над океаном"""
    ocean = _load_ocean_mask(1200, 550)
    if ocean is None:
        raise HTTPException(500, "ocean_mask.npy not found")
    rgba = np.zeros((550, 1200, 4), dtype=np.uint8)
    land = ~ocean
    rgba[land, 0] = 240   # CartoDB light land R
    rgba[land, 1] = 239   # G
    rgba[land, 2] = 234   # B
    rgba[land, 3] = 255   # fully opaque
    from PIL import Image as _PIL
    buf = io.BytesIO()
    _PIL.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/map/{section}/{season_type}/{season_idx}.png")
def api_map_png(section: str, season_type: str, season_idx: int,
                year: Optional[int] = None,
                clim_period: str = "1991_2020",
                percentile: float = 99,
                half: Optional[str] = None,
                scale: int = 1,
                blur: bool = True,
                contours: bool = True,
                contour_labels: bool = False,
                thumb: bool = False):
    if section not in COLORMAPS:
        raise HTTPException(404, "Unknown section")
    if season_type not in ("monthly", "ess", "seasonal", "annual"):
        raise HTTPException(400, "season_type must be monthly|ess|seasonal|annual")
    if half not in (None, "west", "east"):
        raise HTTPException(400, "half must be west|east or omitted")
    scale = max(1, min(scale, 3))
    try:
        png, _, _ = _get_field_png(section, season_type, season_idx, year, clim_period,
                                   percentile, half, scale, blur, contours, contour_labels,
                                   thumb=thumb)
    except HTTPException:
        raise
    except Exception as e:
        import traceback as _tb
        logger.error("Map error %s/%s/%s half=%s: %s\n%s",
                     section, season_type, season_idx, half, e, _tb.format_exc())
        raise HTTPException(500, str(e))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/tile/{section}/{season_type}/{season_idx}/{z}/{x}/{y}.webp")
def api_tile(section: str, season_type: str, season_idx: int,
             z: int, x: int, y: int,
             year: Optional[int] = None,
             clim_period: str = "1991_2020",
             percentile: float = 99,
             blur: bool = True,
             _v: str = ""):
    """256*256 WebP тайл в стандартных XYZ/Slippy координатах (Меркатор с антимеридианом)"""
    if section not in COLORMAPS:
        raise HTTPException(404, "Unknown section")
    if season_type not in ("monthly", "ess", "seasonal", "annual"):
        raise HTTPException(400, "season_type must be monthly|ess|seasonal|annual")
    try:
        tile = _get_tile(section, season_type, season_idx, year, clim_period,
                          percentile, blur, z, x, y)
    except Exception as e:
        import traceback as _tb
        logger.error("Tile error %s/%s/%s z=%d x=%d y=%d: %s\n%s",
                     section, season_type, season_idx, z, x, y, e, _tb.format_exc())
        raise HTTPException(500, str(e))
    return Response(content=tile, media_type="image/webp",
                    headers={"Cache-Control": "public, max-age=3600"})


import base64 as _b64

@app.get("/api/field/{section}/{season_type}/{season_idx}")
def api_field(section: str, season_type: str, season_idx: int,
              year: Optional[int] = None,
              clim_period: str = "1991_2020",
              percentile: float = 99,
              _v: str = ""):
    """Сырые данные поля ERA5 для рендера на клиенте (base64 Float32).
    land_mask=1 - суша ERA5. NaN поля = нет данных в океане (p>=0.05).
    """
    if section not in COLORMAPS:
        raise HTTPException(404, "Unknown section")
    if season_type not in ("monthly", "ess", "seasonal", "annual"):
        raise HTTPException(400, "Bad season_type")
    try:
        field, lat_arr, lon_arr, land_mask, cmap, vmin, vmax = _get_render_context(
            section, season_type, season_idx, year, clim_period, percentile)

        # заполняем сушу для сглаживания на клиенте
        if lat_arr[0] > lat_arr[-1]:
            lat_inc  = lat_arr[::-1]
            field_rc = field[::-1]
            lm_rc    = land_mask[::-1]
        else:
            lat_inc, field_rc, lm_rc = lat_arr, field, land_mask
        field_filled = _fill_land_nans(field_rc, lm_rc)
        # если lat убывал - возвращаем исходный порядок
        if lat_arr[0] > lat_arr[-1]:
            field_filled = field_filled[::-1]

        # кодируем base64 Float32 (NaN сохраняется)
        f32 = field_filled.astype(np.float32)
        field_b64 = _b64.b64encode(f32.ravel().tobytes()).decode()
        lat_b64   = _b64.b64encode(lat_arr.astype(np.float32).tobytes()).decode()
        lon_b64   = _b64.b64encode(lon_arr.astype(np.float32).tobytes()).decode()
        lm_b64    = _b64.b64encode(land_mask.astype(np.uint8).ravel().tobytes()).decode()

        # LUT цветошкалы (256 RGB)
        lut = _LUTS[cmap].tolist()

        return JSONResponse({
            "nlat":       int(len(lat_arr)),
            "nlon":       int(len(lon_arr)),
            "lat":        lat_b64,
            "lon":        lon_b64,
            "field":      field_b64,
            "land_mask":  lm_b64,
            "vmin":       float(vmin),
            "vmax":       float(vmax),
            "lut":        lut,
        }, headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e:
        import traceback as _tb
        logger.error("api_field %s/%s/%s: %s\n%s",
                     section, season_type, season_idx, e, _tb.format_exc())
        raise HTTPException(500, str(e))


@app.get("/api/contours/{section}/{season_type}/{season_idx}.webp")
def api_contours(section: str, season_type: str, season_idx: int,
                 year: Optional[int] = None,
                 clim_period: str = "1991_2020",
                 percentile: float = 99,
                 contour_labels: bool = False,
                 _v: str = ""):
    """Прозрачный WebP только с изолиниями (без заливки), отдельный слой поверх canvas"""
    if section not in COLORMAPS:
        raise HTTPException(404, "Unknown section")
    key = ("contours_only", section, season_type, season_idx, year,
           clim_period, percentile, contour_labels)
    if key in _png_cache:
        return Response(content=_png_cache[key][0], media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=3600"})
    try:
        from PIL import Image as _PIL
        field, lat_arr, lon_arr, land_mask, cmap, vmin, vmax = _get_render_context(
            section, season_type, season_idx, year, clim_period, percentile)

        # репроецируем поле в Меркатор для _overlay_contours
        from scipy.interpolate import RegularGridInterpolator
        W, H = _COAST_W, _COAST_H
        if lat_arr[0] > lat_arr[-1]:
            lat_inc  = lat_arr[::-1]; field_rc = field[::-1]; lm_rc = land_mask[::-1]
        else:
            lat_inc, field_rc, lm_rc = lat_arr, field, land_mask
        field_filled = _fill_land_nans(field_rc, lm_rc)
        dx0, dy0, dx1, dy1 = _MERC_EXTENT
        xs = np.linspace(dx0, dx1, W)
        ys = np.linspace(dy1, dy0, H)
        xs_g, ys_g = np.meshgrid(xs, ys)
        from pyproj import Transformer as _Tr
        _tr2 = _Tr.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lons_f, lats_f = _tr2.transform(xs_g.ravel(), ys_g.ravel())
        lon_min = float(lon_arr.min()); lon_max = float(lon_arr.max())
        lons_n = np.where(lons_f < lon_min, lons_f + 360, lons_f)
        lons_n = np.where(lons_n > lon_max, lons_n - 360, lons_n)
        pts  = np.stack([lats_f, lons_n], axis=1)
        interp = RegularGridInterpolator(
            (lat_inc, lon_arr), field_filled,
            method="linear", bounds_error=False, fill_value=np.nan)
        vals   = interp(pts).reshape(H, W)
        finite = np.isfinite(vals)
        ocean_mask_s = _load_ocean_mask(W, H)

        base = _PIL.new("RGBA", (W, H), (0, 0, 0, 0))
        img  = _overlay_contours(base, vals, finite, vmin, vmax, ocean_mask_s,
                                  show_labels=contour_labels, lut=_LUTS[cmap])
        buf = io.BytesIO()
        img.save(buf, format="WEBP", lossless=True, quality=80)
        result = buf.getvalue()
        _png_cache[key] = (result, float(vmin), float(vmax))
        return Response(content=result, media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e:
        import traceback as _tb
        logger.error("api_contours %s/%s/%s: %s\n%s",
                     section, season_type, season_idx, e, _tb.format_exc())
        raise HTTPException(500, str(e))


_coastline_cache: dict = {}

# вспомогательные функции для эндпоинтов береговой линии
_COAST_LON_MIN, _COAST_LON_MAX = 117.0, 246.0
_COAST_LAT_MIN, _COAST_LAT_MAX = 26.0,  72.0
_COAST_W, _COAST_H             = 1200,  550
_EARTH_CIRC_COAST               = 40075016.685578488

def _coast_boxes():
    from shapely.geometry import box as _box
    # внешние поля для обрезки полигонов; большой overlap антимеридиана (5) устраняет
    # видимый шов на 180 - полигоны на обе стороны от антимеридиана рендерятся дважды
    M_LEFT  = 30.0   # 117 - 30 = 87°E  (west clip)
    M_RIGHT = 20.0   # 246 + 20 = 266°E (east clip, in −lon terms: -114 + 20 = -94°W)
    M_SOUTH = 25.0   # 26  - 25 =  1°N  (south clip)
    M_NORTH =  5.0   # 72  +  5 = 77°N  (north clip)
    M_ANTI  =  5.0   # antimeridian overlap
    return (
        _box(_COAST_LON_MIN - M_LEFT,  _COAST_LAT_MIN - M_SOUTH,
             180.0 + M_ANTI,           _COAST_LAT_MAX + M_NORTH),
        _box(-180.0 - M_ANTI,          _COAST_LAT_MIN - M_SOUTH,
             -(360.0 - _COAST_LON_MAX) + M_RIGHT,  _COAST_LAT_MAX + M_NORTH),
    )

def _coast_projection():
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, _ = tr.transform(_COAST_LON_MIN, _COAST_LAT_MIN)
    x1    = x0 + (_COAST_LON_MAX - _COAST_LON_MIN) / 360.0 * _EARTH_CIRC_COAST
    _, y1 = tr.transform(0.0, _COAST_LAT_MAX)
    _, y0 = tr.transform(0.0, _COAST_LAT_MIN)
    return tr, x0, x1, y0, y1

def _coast_to_px(lon_std, lat, east, tr, x0, x1, y0, y1):
    mx, my = tr.transform(lon_std, lat)
    if east:
        mx += _EARTH_CIRC_COAST
    px = (mx - x0) / (x1 - x0) * _COAST_W
    py = (y1 - my) / (y1 - y0) * _COAST_H
    return px, py

def _load_coast_geojson():
    import json
    path = os.path.join(DATA_DIR, "ne_10m_land.geojson")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/coastline.svg")
def api_coastline_svg():
    """Векторная береговая линия спроецированная в пиксельное пространство 1200*550"""
    if "svg" in _coastline_cache:
        return Response(content=_coastline_cache["svg"], media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})
    from shapely.geometry import shape
    from shapely.validation import make_valid

    geo = _load_coast_geojson()
    if geo is None:
        raise HTTPException(404, "ne_10m_land.geojson not found — run precompute_coastline.py")

    BOX_W, BOX_E = _coast_boxes()
    tr, x0, x1, y0, y1 = _coast_projection()

    def ring_to_path(coords, east):
        pts = []
        prev_px = None
        for lon, lat in coords:
            try:
                px, py = _coast_to_px(lon, lat, east, tr, x0, x1, y0, y1)
            except Exception:
                continue
            if prev_px is not None and abs(px - prev_px) > _COAST_W * 0.3:
                pts.append(None)   # antimeridian break
            pts.append((px, py))
            prev_px = px
        if not pts:
            return ""
        segments, cur = [], []
        for p in pts:
            if p is None:
                if len(cur) >= 2:
                    segments.append(cur)
                cur = []
            else:
                cur.append(p)
        if len(cur) >= 2:
            segments.append(cur)
        d = ""
        for seg in segments:
            d += f"M{seg[0][0]:.1f},{seg[0][1]:.1f}"
            for px, py in seg[1:]:
                d += f"L{px:.1f},{py:.1f}"
        return d

    def geom_to_path(geom, east):
        polys = [geom] if geom.geom_type == "Polygon" else (
            [g for g in geom.geoms if g.geom_type == "Polygon"]
            if geom.geom_type in ("MultiPolygon", "GeometryCollection") else [])
        d = ""
        for poly in polys:
            if poly.is_empty:
                continue
            d += ring_to_path(poly.exterior.coords, east)
            for hole in poly.interiors:
                d += ring_to_path(hole.coords, east)
        return d

    all_d = ""
    for feat in geo["features"]:
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        geom = geom.simplify(0.08, preserve_topology=True)
        w = geom.intersection(BOX_W)
        if not w.is_empty:
            all_d += geom_to_path(w, False)
        e = geom.intersection(BOX_E)
        if not e.is_empty:
            all_d += geom_to_path(e, True)

    W, H = _COAST_W, _COAST_H
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
        f'<defs>'
        f'<filter id="bf" x="-2%" y="-2%" width="104%" height="104%">'
        f'<feGaussianBlur stdDeviation="0.7"/>'
        f'</filter>'
        f'</defs>'
        f'<path d="{all_d}" fill="none" filter="url(#bf)" '
        f'stroke="rgba(20,30,60,0.18)" stroke-width="2.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<path d="{all_d}" fill="none" '
        f'stroke="rgba(15,25,55,0.52)" stroke-width="0.7" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )
    data = svg.encode("utf-8")
    _coastline_cache["svg"] = data
    return Response(content=data, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/coastline.geojson")
def api_coastline_geojson():
    """Полигоны Natural Earth обрезанные по домену, разбитые west/east для OL"""
    if "geojson" in _coastline_cache:
        return Response(content=_coastline_cache["geojson"],
                        media_type="application/geo+json",
                        headers={"Cache-Control": "public, max-age=86400"})
    import json
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid

    geo = _load_coast_geojson()
    if geo is None:
        raise HTTPException(404, "ne_10m_land.geojson not found")

    BOX_W, BOX_E = _coast_boxes()
    features = []
    for feat in geo["features"]:
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom)
        geom = geom.simplify(0.08, preserve_topology=True)
        w = geom.intersection(BOX_W)
        if not w.is_empty:
            features.append({"type": "Feature", "geometry": mapping(w),
                              "properties": {"half": "west"}})
        e = geom.intersection(BOX_E)
        if not e.is_empty:
            features.append({"type": "Feature", "geometry": mapping(e),
                              "properties": {"half": "east"}})

    data = json.dumps({"type": "FeatureCollection", "features": features},
                      separators=(',', ':')).encode()
    _coastline_cache["geojson"] = data
    return Response(content=data, media_type="application/geo+json",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/colorscale/{section}")
def api_colorscale(section: str,
                   season_type: str = "monthly",
                   season_idx: int = 0,
                   year: Optional[int] = None,
                   clim_period: str = "1991_2020",
                   percentile: float = 99,
                   half: Optional[str] = None):
    if section not in COLORMAPS:
        raise HTTPException(404, "Unknown section")
    _, vmin, vmax = _get_field_png(section, season_type, season_idx, year, clim_period, percentile, half)
    cmap, _  = COLORMAPS[section]
    unit     = next((s["unit"] for s in SECTIONS if s["id"] == section), "")
    rng      = vmax - vmin

    import math as _math
    tick_vals = _get_tick_values(section)
    # фильтр тиков по dedup нормы
    filtered_vals, filtered_norms = [], []
    for v in tick_vals:
        v_f  = float(v)
        norm = (v_f - vmin) / rng if rng != 0 else 0.5
        norm = max(0.0, min(1.0, norm))
        if filtered_norms and abs(norm - filtered_norms[-1]) < 0.02:
            continue
        filtered_vals.append(v_f)
        filtered_norms.append(norm)

    # оптимальная точность по мин. шагу между соседними значениями
    if len(filtered_vals) >= 2:
        steps = [abs(filtered_vals[i+1] - filtered_vals[i])
                 for i in range(len(filtered_vals) - 1)]
        min_step = min(s for s in steps if s > 0) if any(s > 0 for s in steps) else rng
    else:
        min_step = rng if rng > 0 else 1.0
    if min_step > 0:
        prec = max(0, min(4, _math.ceil(-_math.log10(min_step))))
    else:
        prec = 2

    ticks = [
        {
            "value": round(v_f, 4),
            "label": _fmt_tick_prec(v_f, prec),
            "norm":  round(norm, 4),
        }
        for v_f, norm in zip(filtered_vals, filtered_norms)
    ]

    return {
        "vmin":  round(vmin, 4),
        "vmax":  round(vmax, 4),
        "unit":  unit,
        "cmap":  cmap,
        "lut":   _LUTS[cmap].tolist(),
        "ticks": ticks,
    }


@app.get("/api/timeseries")
def api_timeseries(lat: float, lon: float):
    meta    = _load("meta")
    lat_arr = meta["lat"]
    lon_arr = meta["lon"]
    i = int(np.argmin(np.abs(lat_arr - lat)))
    j = int(np.argmin(np.abs(lon_arr - lon)))
    sst     = _load("sst_raw")["sst"]
    ts      = sst[:, i, j].tolist()
    dates   = [str(d) for d in meta["date_strings"]]
    return {
        "lat":   float(lat_arr[i]),
        "lon":   float(lon_arr[j]),
        "dates": dates,
        "sst":   [round(float(v), 3) if math.isfinite(float(v)) else None for v in ts],
    }


@app.get("/api/anomaly/{year}/{month}")
def api_anomaly(year: int, month: int, clim_period: str = "1991_2020"):
    meta      = _load("meta")
    years_a   = meta["years"]
    months_a  = meta["months"]
    land_mask = meta["land_mask"]
    raw_sst   = _load("sst_raw")["sst"]
    s7        = _load("s7")
    clim_key  = f"clim_{clim_period}"
    if clim_key not in s7:
        raise HTTPException(400, f"Unknown clim_period: {clim_period}")
    clim = s7[clim_key]
    idx  = np.where((years_a == year) & (months_a == month))[0]
    if len(idx) == 0:
        raise HTTPException(404, f"No data for {year}-{month:02d}")
    field = raw_sst[idx[0]].astype(np.float32) - clim[month - 1]
    field[land_mask] = np.nan
    valid = field[np.isfinite(field)]
    return {
        "year": year, "month": month,
        "clim_period": clim_period,
        "mean":    round(float(np.nanmean(valid)), 3),
        "max":     round(float(np.nanmax(valid)),  3),
        "min":     round(float(np.nanmin(valid)),  3),
        "anom_vm": float(meta["anom_vm"]),
    }


@app.get("/api/field_value")
def api_field_value(section: str, lat: float, lon: float,
                    season_type: str = "monthly", season_idx: int = 0,
                    year: Optional[int] = None,
                    clim_period: str = "1991_2020"):
    meta      = _load("meta")
    lat_arr   = meta["lat"]
    lon_arr   = meta["lon"]
    land_mask = meta["land_mask"]
    i = int(np.argmin(np.abs(lat_arr - lat)))
    j = int(np.argmin(np.abs(lon_arr - lon)))
    unit = next((s["unit"] for s in SECTIONS if s["id"] == section), "")
    if land_mask[i, j]:
        return {"value": None, "unit": unit}
    if section == "s7":
        if year is None:
            year = int(meta["years"][-1])
        field = _compute_s7_field(season_type, season_idx, year, clim_period)
        value = float(field[i, j])
    elif section == "s5" and season_type == "monthly":
        meta_s5 = _load("meta")
        field   = _get_s5_monthly_field(season_idx, meta_s5)
        value   = float(field[i, j])
    elif season_type in ("monthly", "ess"):
        key2 = "monthly" if season_type == "monthly" else "ess"
        data  = _load(section)
        arr   = data[key2]
        value = float(arr[season_idx][i, j])
    else:
        data  = _load(section)
        value = float(data["monthly"][season_idx][i, j])
    if not math.isfinite(value):
        return {"value": None, "unit": unit}
    return {"value": round(value, 3), "unit": unit}


@app.get("/api/point_stats")
def api_point_stats(lat: float, lon: float):
    meta      = _load("meta")
    lat_arr   = meta["lat"]
    lon_arr   = meta["lon"]
    land_mask = meta["land_mask"]
    i = int(np.argmin(np.abs(lat_arr - lat)))
    j = int(np.argmin(np.abs(lon_arr - lon)))
    if land_mask[i, j]:
        return {"land": True}
    s2 = _load("s2")
    s3 = _load("s3")
    trends = [float(s2["monthly"][m][i, j]) for m in range(12)]
    r2s    = [float(s3["monthly"][m][i, j]) for m in range(12)]
    valid_t = [v for v in trends if math.isfinite(v)]
    valid_r = [v for v in r2s    if math.isfinite(v)]
    return {
        "lat":          round(float(lat_arr[i]), 3),
        "lon":          round(float(lon_arr[j]), 3),
        "trend_per_dec": round(float(np.mean(valid_t)) if valid_t else 0.0, 4),
        "r2":            round(float(np.mean(valid_r)) if valid_r else 0.0, 4),
    }


@app.get("/api/point_series")
def api_point_series(section: str, lat: float, lon: float,
                     clim: str = "1991_2020"):
    """Return a time-series for the nearest grid point suitable for Chart.js."""
    meta      = _load("meta")
    lat_arr   = meta["lat"]
    lon_arr   = meta["lon"]
    land_mask = meta["land_mask"]
    i = int(np.argmin(np.abs(lat_arr - lat)))
    j = int(np.argmin(np.abs(lon_arr - lon)))

    if land_mask[i, j]:
        return {"labels": [], "values": [], "unit": "", "chart_type": "line"}

    def _safe(v: float) -> Optional[float]:
        return round(v, 4) if math.isfinite(v) else None

    if section == "s1":
        data   = _load("s1")
        values = [_safe(float(data["monthly"][m][i, j])) for m in range(12)]
        return {"labels": MON_LABELS, "values": values, "unit": "°C", "chart_type": "line"}

    elif section == "s2":
        data   = _load("s2")
        values = [_safe(float(data["monthly"][m][i, j])) for m in range(12)]
        return {"labels": MON_LABELS, "values": values, "unit": "°C/дек", "chart_type": "bar"}

    elif section == "s3":
        data   = _load("s3")
        values = [_safe(float(data["monthly"][m][i, j])) for m in range(12)]
        return {"labels": MON_LABELS, "values": values, "unit": "R²", "chart_type": "line"}

    elif section == "s4":
        # годовой ряд дисперсии
        raw     = _load("sst_raw")["sst"]
        years_a = meta["years"]
        all_yrs = sorted(set(int(y) for y in years_a))
        values  = []
        for yr in all_yrs:
            ix = np.where(years_a == yr)[0]
            v  = float(np.nanvar(raw[ix, i, j])) if len(ix) else float("nan")
            values.append(_safe(v))
        return {"labels": [str(y) for y in all_yrs], "values": values,
                "unit": "°C²", "chart_type": "line"}

    elif section == "s5":
        # тренд дисперсии по месяцам
        values = []
        for m in range(12):
            f = _get_s5_monthly_field(m, meta)
            values.append(_safe(float(f[i, j])))
        return {"labels": MON_LABELS, "values": values,
                "unit": "°C²/дек", "chart_type": "bar"}

    elif section == "s7":
        # полный помесячный ряд аномалий
        raw      = _load("sst_raw")["sst"]
        s7       = _load("s7")
        clim_key = f"clim_{clim}"
        if clim_key not in s7:
            raise HTTPException(400, f"Unknown clim period: {clim}")
        clim_data = s7[clim_key]
        years_a   = meta["years"]
        months_a  = meta["months"]
        labels, values = [], []
        for k in range(len(years_a)):
            yr, mo = int(years_a[k]), int(months_a[k])
            labels.append(f"{yr}-{mo:02d}")
            raw_v  = float(raw[k, i, j])
            clim_v = float(clim_data[mo - 1, i, j])
            values.append(_safe(raw_v - clim_v))
        return {"labels": labels, "values": values, "unit": "°C", "chart_type": "line"}

    raise HTTPException(404, f"No point_series for section {section}")


@app.get("/api/point_annual")
def api_point_annual(section: str, lat: float, lon: float,
                     clim: str = "1991_2020",
                     month: Optional[int] = None,   # 0-based month index (0=Jan), None=annual
                     ess: Optional[int] = None):     # 0-3 season, overrides month if set
    """Annual time series for a point: raw SST (s1) or anomaly (s7), by year."""
    meta      = _load("meta")
    lat_arr   = meta["lat"]
    lon_arr   = meta["lon"]
    land_mask = meta["land_mask"]
    i = int(np.argmin(np.abs(lat_arr - lat)))
    j = int(np.argmin(np.abs(lon_arr - lon)))
    if land_mask[i, j]:
        return {"labels": [], "values": [], "unit": ""}

    raw_sst = _load("sst_raw")["sst"]
    years_a  = meta["years"]
    months_a = meta["months"]
    all_years = sorted(set(int(y) for y in years_a))

    def _safe(v: float):
        return round(float(v), 3) if math.isfinite(float(v)) else None

    def _get_indices(yr: int):
        """Индексы времени для года и выбранного месяца/сезона"""
        if ess is not None:
            mons_in_season = _SEASONAL_MONTHS[ess % 4]
            idx = []
            for mo in mons_in_season:
                mo1  = (mo % 12) + 1
                yr_  = yr if mo < 12 else yr - 1
                ix   = np.where((years_a == yr_) & (months_a == mo1))[0]
                idx.extend(ix.tolist())
            return np.array(idx, dtype=int)
        elif month is not None:
            return np.where((years_a == yr) & (months_a - 1 == month))[0]
        else:
            return np.where(years_a == yr)[0]

    if section == "s1":
        labels, values = [], []
        for yr in all_years:
            ix = _get_indices(yr)
            v  = np.nanmean(raw_sst[ix, i, j]) if len(ix) else float("nan")
            labels.append(str(yr))
            values.append(_safe(v))
        return {"labels": labels, "values": values, "unit": "°C"}

    elif section == "s7":
        s7       = _load("s7")
        clim_key = f"clim_{clim}"
        if clim_key not in s7:
            raise HTTPException(400, f"Unknown clim: {clim}")
        clim_data = s7[clim_key]   # (12, NLAT, NLON)

        labels, values = [], []
        for yr in all_years:
            ix = _get_indices(yr)
            if not len(ix):
                labels.append(str(yr)); values.append(None); continue
            raw_vals  = [float(raw_sst[k, i, j]) for k in ix]
            # климатология для каждого шага
            clim_vals = [float(clim_data[int(months_a[k]) - 1, i, j]) for k in ix]
            anoms     = [r - c for r, c in zip(raw_vals, clim_vals)
                         if math.isfinite(r) and math.isfinite(c)]
            v = float(np.mean(anoms)) if anoms else float("nan")
            labels.append(str(yr))
            values.append(_safe(v))
        return {"labels": labels, "values": values, "unit": "°C"}

    raise HTTPException(404, f"point_annual not supported for {section}")


# статика + index
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def _prewarm_cache():
    """Предварительный рендер всех месячных карт для дефолтной секции в фоне"""
    import threading

    def _warm():
        try:
            for sec in list(COLORMAPS.keys()):
                for mo in range(12):
                    try:
                        # полный рендер (для обзора)
                        _get_field_png(sec, "monthly", mo, None, "1991_2020",
                                       99, None, 1, True, True, False)
                        # кэш контекста (field + vmin/vmax) для тайлов
                        _get_render_context(sec, "monthly", mo, None, "1991_2020", 99)
                    except Exception as e:
                        logger.debug("Prewarm %s/monthly/%d: %s", sec, mo, e)
            logger.info("Cache pre-warm complete (img=%d, ctx=%d)",
                        len(_png_cache), len(_field_render_cache))
        except Exception as e:
            logger.warning("Cache pre-warm failed: %s", e)

    threading.Thread(target=_warm, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.11.213.22", port=8080)
