#!/usr/bin/env python3
"""
Скрипт подготовки данных.

  1. SST   (data/sst_era5.nc)        — полная закачка или докачка месяцев с CDS
  2. T2M   (data/t2m_era5.nc)        — то же
  3. Coast (data/ocean_mask*.npy)    — растеризация Natural Earth
  4. NPZ   (data/{,t2m_}{meta,s1..s7,sst_raw}.npz) — precompute.py --var

Использование (CLI):
  python prepare_data.py                  # проверить и обновить всё
  python prepare_data.py --check          # только проверить (exit 1 если чего-то нет)
  python prepare_data.py --skip-t2m       # пропустить T2M
  python prepare_data.py --skip-download  # без обращения к CDS (только пересчёт)
  python prepare_data.py --force          # принудительно пересчитать NPZ
  python prepare_data.py --lag 3          # задержка CDS 3 месяца

Использование (импорт):
  from prepare_data import data_status, ensure_data
  ok, missing = data_status()             # быстрая проверка
  ensure_data(skip_download=True)          # без CDS — только локальный пересчёт

Требования для CDS:
  ~/.cdsapirc с url и key (см. https://cds.climate.copernicus.eu)
  pip: cdsapi, netCDF4, numpy, scipy, Pillow, pyproj, shapely
"""
from __future__ import annotations
import os, sys, argparse, shutil, subprocess, logging
from datetime import datetime, date
from typing import Optional
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SST_NC = os.path.join(DATA_DIR, "sst_era5.nc")
T2M_NC = os.path.join(DATA_DIR, "t2m_era5.nc")

CDS_DATASET    = "reanalysis-era5-single-levels-monthly-means"
CDS_AREA       = [72, 117, 26, 246]      # [N, W, S, E]
CDS_LAG_MONTHS = 2                        # ERA5T публикуется с задержкой ~2 мес

# файлы по которым проверяется готовность
SST_NPZ_FILES = ["meta.npz", "sst_raw.npz",
                 "s1.npz", "s2.npz", "s3.npz", "s4.npz", "s5.npz", "s6.npz", "s7.npz"]
T2M_NPZ_FILES = ["t2m_meta.npz", "t2m_raw.npz",
                 "t2m_s1.npz", "t2m_s2.npz", "t2m_s3.npz", "t2m_s4.npz",
                 "t2m_s5.npz", "t2m_s6.npz", "t2m_s7.npz"]
COAST_FILES   = ["ocean_mask.npy", "ocean_mask_hires.npy", "ne_10m_land.geojson"]

logger = logging.getLogger(__name__)

# Проверка готовности

def data_status(require_t2m: bool = True) -> tuple[bool, list[str]]:
    """Возвращает (ok, missing). ok=True если всё нужное есть для запуска сервера."""
    missing: list[str] = []
    needed = list(SST_NPZ_FILES) + COAST_FILES
    if require_t2m:
        needed += T2M_NPZ_FILES
    for f in needed:
        if not os.path.exists(os.path.join(DATA_DIR, f)):
            missing.append(f)
    return (len(missing) == 0, missing)


# Скачивание ERA5

def _last_nc_date(path: str) -> datetime:
    import netCDF4
    with netCDF4.Dataset(path, "r") as nc:
        vt = np.array(nc.variables["valid_time"][:])
    return datetime.utcfromtimestamp(int(np.nanmax(vt)))


def _cds_latest(lag: int) -> tuple[int, int]:
    today = date.today()
    m = today.month - lag
    y = today.year
    while m <= 0:
        m += 12; y -= 1
    return y, m


def _missing_months(last: datetime, up_y: int, up_m: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = last.year, last.month
    m += 1
    if m > 12:
        m, y = 1, y + 1
    while (y, m) <= (up_y, up_m):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _nc_time_is_unlimited(path: str) -> bool:
    import netCDF4
    with netCDF4.Dataset(path, "r") as nc:
        for dim in nc.dimensions.values():
            if dim.name == "valid_time" and dim.isunlimited():
                return True
    return False


def _download_cds(variable: str, years: list[str], months: list[str],
                  tmp_path: str, label: str) -> None:
    import cdsapi
    req = {
        "product_type":    ["monthly_averaged_reanalysis"],
        "variable":        [variable],
        "year":            years,
        "month":           months,
        "time":            ["00:00"],
        "data_format":     "netcdf",
        "download_format": "unarchived",
        "area":            CDS_AREA,
    }
    print(f"  CDS: скачиваем {label} ...")
    cdsapi.Client().retrieve(CDS_DATASET, req).download(tmp_path)
    print(f"  → {tmp_path}")


def _append_nc(dst_path: str, src_path: str, var: str) -> int:
    import netCDF4
    with netCDF4.Dataset(src_path, "r") as src:
        src_vt  = np.array(src.variables["valid_time"][:])
        src_dat = np.array(src.variables[var][:])
    with netCDF4.Dataset(dst_path, "r") as dst:
        dst_vt_set = set(int(v) for v in dst.variables["valid_time"][:])
        t_cur = len(dst.variables["valid_time"][:])
    new_mask = np.array([int(v) not in dst_vt_set for v in src_vt])
    if not new_mask.any():
        return 0
    new_vt  = src_vt[new_mask]
    new_dat = src_dat[new_mask]
    order   = np.argsort(new_vt)
    new_vt  = new_vt[order]
    new_dat = new_dat[order]
    n = len(new_vt)
    with netCDF4.Dataset(dst_path, "a") as dst:
        dst.variables["valid_time"][t_cur:t_cur + n] = new_vt
        dst.variables[var][t_cur:t_cur + n]          = new_dat
    return n


def _merge_to_new_file(dst_path: str, src_path: str, var: str) -> int:
    import netCDF4
    print("  valid_time не UNLIMITED - созд. merged файл...")
    tmp_merged = dst_path + ".merged.nc"
    with netCDF4.Dataset(dst_path, "r") as old, \
         netCDF4.Dataset(src_path, "r") as src, \
         netCDF4.Dataset(tmp_merged, "w", format="NETCDF4") as out:
        old_vt  = np.array(old.variables["valid_time"][:])
        old_dat = np.array(old.variables[var][:])
        src_vt  = np.array(src.variables["valid_time"][:])
        src_dat = np.array(src.variables[var][:])
        old_vt_set = set(int(v) for v in old_vt)
        new_mask = np.array([int(v) not in old_vt_set for v in src_vt])
        n = int(new_mask.sum())
        if n == 0:
            return 0
        all_vt  = np.concatenate([old_vt, src_vt[new_mask]])
        all_dat = np.concatenate([old_dat, src_dat[new_mask]])
        order   = np.argsort(all_vt)
        all_vt  = all_vt[order]
        all_dat = all_dat[order]
        out.createDimension("valid_time", None)
        out.createDimension("latitude",  len(old.variables["latitude"][:]))
        out.createDimension("longitude", len(old.variables["longitude"][:]))
        for vname in ("latitude", "longitude"):
            v_in  = old.variables[vname]
            v_out = out.createVariable(vname, v_in.dtype, v_in.dimensions)
            v_out.setncatts({k: v_in.getncattr(k) for k in v_in.ncattrs()})
            v_out[:] = v_in[:]
        vt_out = out.createVariable("valid_time", "i8", ("valid_time",))
        try:
            vt_out.setncatts({k: old.variables["valid_time"].getncattr(k)
                              for k in old.variables["valid_time"].ncattrs()})
        except Exception:
            pass
        vt_out[:] = all_vt
        dat_out = out.createVariable(var, "f4",
                                     ("valid_time", "latitude", "longitude"),
                                     zlib=True, complevel=4)
        try:
            dat_out.setncatts({k: old.variables[var].getncattr(k)
                               for k in old.variables[var].ncattrs()})
        except Exception:
            pass
        dat_out[:] = all_dat
    shutil.move(tmp_merged, dst_path)
    return n


def _ensure_nc(nc_path: str, cds_var: str, label: str, lag: int) -> bool:
    """Скачать или докачать NC. Возвращает True если файл обновился."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = nc_path + ".download.nc"
    updated = False
    try:
        if not os.path.exists(nc_path):
            print(f"[{label}] Файл не найден - полная загрузка с CDS...")
            up_y, up_m = _cds_latest(lag)
            years  = [str(y) for y in range(1979, up_y + 1)]
            months = [f"{m:02d}" for m in range(1, 13)]
            _download_cds(cds_var, years, months, tmp,
                          f"1979-01 .. {up_y}-{up_m:02d}")
            shutil.move(tmp, nc_path)
            print(f"  Сохранено: {nc_path}")
            updated = True
        else:
            last       = _last_nc_date(nc_path)
            up_y, up_m = _cds_latest(lag)
            print(f"[{label}] Последний шаг: {last.year}-{last.month:02d}  "
                  f"доступно на CDS: {up_y}-{up_m:02d}")
            missing = _missing_months(last, up_y, up_m)
            if not missing:
                print("  Данные актуальны")
                return False
            print(f"  Новых месяцев: {len(missing)}")
            yrs = sorted(set(str(y) for y, _ in missing))
            mos = sorted(set(f"{m:02d}" for _, m in missing))
            _download_cds(cds_var, yrs, mos, tmp,
                          f"{missing[0][0]}-{missing[0][1]:02d}"
                          f" .. {missing[-1][0]}-{missing[-1][1]:02d}")
            if _nc_time_is_unlimited(nc_path):
                added = _append_nc(nc_path, tmp, cds_var)
            else:
                added = _merge_to_new_file(nc_path, tmp, cds_var)
            if added > 0:
                last_new = _last_nc_date(nc_path)
                print(f"  Добавлено {added} шагов. Последний: "
                      f"{last_new.year}-{last_new.month:02d}")
                updated = True
            else:
                print("  Новых данных не найдено (возможно, дубликаты)")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return updated


# Береговая линия

def _ensure_coastline(force: bool = False) -> bool:
    coast_exists = all(
        os.path.exists(os.path.join(DATA_DIR, f)) for f in COAST_FILES
    )
    if coast_exists and not force:
        print("[Coastline] Маска суши и geojson на месте, пропуск")
        return False
    print("[Coastline] Растеризация Natural Earth ...")
    script = os.path.join(BASE_DIR, "precompute_coastline.py")
    subprocess.run([sys.executable, script], check=True, cwd=BASE_DIR)
    return True


# NPZ предвычисление

def _npz_need_update(nc_path: str, key_file: str, force: bool) -> bool:
    if force:
        return True
    if not os.path.exists(key_file):
        return True
    if not os.path.exists(nc_path):
        return False
    return os.path.getmtime(nc_path) > os.path.getmtime(key_file)


def _ensure_precomputed(nc_path: str, var: str, force: bool = False) -> bool:
    key_file = os.path.join(DATA_DIR, "meta.npz" if var == "sst" else f"{var}_meta.npz")
    if not os.path.exists(nc_path):
        print(f"[Precompute/{var}] NC файл отсутствует, пропуск")
        return False
    if not _npz_need_update(nc_path, key_file, force):
        print(f"[Precompute/{var}] NPZ актуальны, пропуск")
        return False
    print(f"[Precompute/{var}] Запуск precompute.py --var {var}...")
    script = os.path.join(BASE_DIR, "precompute.py")
    subprocess.run([sys.executable, script, "--var", var], check=True, cwd=BASE_DIR)
    return True


# Оркестратор

def ensure_data(skip_sst: bool        = False,
                skip_t2m: bool        = False,
                skip_coastline: bool  = False,
                skip_precompute: bool = False,
                skip_download: bool   = False,
                force: bool           = False,
                lag: int              = CDS_LAG_MONTHS) -> bool:
    """Проверить и при необходимости скачать/пересчитать всё"""
    ok = True
    sst_updated = False

    if not skip_sst and not skip_download:
        try:
            sst_updated = _ensure_nc(SST_NC, "sea_surface_temperature", "SST", lag)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ОШИБКА [SST]: {e}", file=sys.stderr); ok = False

    if not skip_t2m and not skip_download:
        try:
            _ensure_nc(T2M_NC, "2m_temperature", "T2M", lag)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ОШИБКА [T2M]: {e}", file=sys.stderr); ok = False

    if not skip_coastline:
        try:
            _ensure_coastline()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ОШИБКА [Coastline]: {e}", file=sys.stderr); ok = False

    if not skip_precompute:
        if not skip_sst:
            try:
                _ensure_precomputed(SST_NC, "sst", force=force or sst_updated)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  ОШИБКА [Precompute/sst]: {e}", file=sys.stderr); ok = False
        if not skip_t2m:
            try:
                _ensure_precomputed(T2M_NC, "t2m", force=force)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  ОШИБКА [Precompute/t2m]: {e}", file=sys.stderr); ok = False

    return ok


# CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check",           action="store_true",
                   help="Только проверить наличие файлов; exit 1 если чего-то нет")
    p.add_argument("--skip-sst",        action="store_true")
    p.add_argument("--skip-t2m",        action="store_true")
    p.add_argument("--skip-coastline",  action="store_true")
    p.add_argument("--skip-precompute", action="store_true")
    p.add_argument("--skip-download",   action="store_true",
                   help="Не обращаться к CDS - только локальный пересчёт")
    p.add_argument("--force",           action="store_true",
                   help="Принудительно пересчитать NPZ даже если актуальны")
    p.add_argument("--lag",             type=int, default=CDS_LAG_MONTHS)
    args = p.parse_args()

    if args.check:
        ok, missing = data_status(require_t2m=not args.skip_t2m)
        if ok:
            print("data_status: OK - все файлы на месте")
            sys.exit(0)
        print("data_status: MISSING ({} файлов):".format(len(missing)))
        for f in missing:
            print(f"  - data/{f}")
        sys.exit(1)

    ok = ensure_data(
        skip_sst        = args.skip_sst,
        skip_t2m        = args.skip_t2m,
        skip_coastline  = args.skip_coastline,
        skip_precompute = args.skip_precompute,
        skip_download   = args.skip_download,
        force           = args.force,
        lag             = args.lag,
    )
    print("\nГотово!" if ok else "\nЗавершено с ошибками (см. выше)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
