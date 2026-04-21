#!/usr/bin/env python3
"""
Проверка и докачка новых данных ERA5 SST с CDS.
Запускать раз в месяц. После успешного обновления - python precompute.py.

Требования: учётная запись на https://cds.climate.copernicus.eu
             ~/.cdsapirc с url и key (см. README)
"""

import os, sys, argparse
from datetime import datetime, date
import numpy as np

NC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sst_era5.nc")

CDS_DATASET = "reanalysis-era5-single-levels-monthly-means"
CDS_AREA    = [72, 117, 26, 246]   # [N, W, S, E]

# ERA5 MODA публикуется с задержкой ~2 месяца (ERA5T - предварительные данные)
CDS_LAG_MONTHS = 2


def _last_nc_date(path: str) -> datetime:
    """Последний временной шаг в NC файле"""
    import netCDF4
    with netCDF4.Dataset(path, "r") as nc:
        vt = np.array(nc.variables["valid_time"][:])
    return datetime.utcfromtimestamp(int(np.nanmax(vt)))


def _cds_latest() -> tuple[int, int]:
    """Последний месяц доступный на CDS (текущий - LAG)"""
    today = date.today()
    m = today.month - CDS_LAG_MONTHS
    y = today.year
    while m <= 0:
        m += 12
        y -= 1
    return y, m


def _missing_months(last: datetime, up_y: int, up_m: int) -> list[tuple[int, int]]:
    """Список (year, month) отсутствующих в NC"""
    result = []
    y, m = last.year, last.month
    m += 1
    if m > 12:
        m, y = 1, y + 1
    while (y, m) <= (up_y, up_m):
        result.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return result


def _download(months: list[tuple[int, int]], tmp_path: str) -> None:
    """Скачать указанные месяцы с CDS"""
    import cdsapi

    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable":     ["sea_surface_temperature"],
        "year":         sorted(set(str(y)        for y, _ in months)),
        "month":        sorted(set(f"{m:02d}"    for _, m in months)),
        "time":         ["00:00"],
        "data_format":  "netcdf",
        "download_format": "unarchived",
        "area":         CDS_AREA,
    }

    print(f"CDS: скачиваем {len(months)} месяц(а): "
          f"{months[0][0]}-{months[0][1]:02d} .. {months[-1][0]}-{months[-1][1]:02d}")
    cdsapi.Client().retrieve(CDS_DATASET, req).download(tmp_path)
    print(f"Загружено -> {tmp_path}")


def _nc_time_is_unlimited(path: str) -> bool:
    import netCDF4
    with netCDF4.Dataset(path, "r") as nc:
        for dim in nc.dimensions.values():
            if dim.name == "valid_time" and dim.isunlimited():
                return True
    return False


def _append_nc(dst_path: str, src_path: str) -> int:
    """Добавить новые временные шаги из src в dst. Вернуть кол-во добавленных"""
    import netCDF4

    with netCDF4.Dataset(src_path, "r") as src:
        src_vt  = np.array(src.variables["valid_time"][:])
        src_sst = np.array(src.variables["sst"][:])

    with netCDF4.Dataset(dst_path, "r") as dst:
        dst_vt_set = set(int(v) for v in dst.variables["valid_time"][:])
        t_cur = len(dst.variables["valid_time"][:])

    # исключаем дубликаты
    new_mask = np.array([int(v) not in dst_vt_set for v in src_vt])
    if not new_mask.any():
        return 0

    new_vt  = src_vt[new_mask]
    new_sst = src_sst[new_mask]

    # сортировка по времени
    order   = np.argsort(new_vt)
    new_vt  = new_vt[order]
    new_sst = new_sst[order]

    n = len(new_vt)
    with netCDF4.Dataset(dst_path, "a") as dst:
        dst.variables["valid_time"][t_cur:t_cur + n] = new_vt
        dst.variables["sst"][t_cur:t_cur + n]        = new_sst

    return n


def _merge_to_new_file(dst_path: str, src_path: str) -> int:
    """Фолбэк: создать новый merged файл (если время не UNLIMITED в dst)"""
    import netCDF4, shutil

    print("valid_time не UNLIMITED - создаём новый merged файл...")
    tmp_merged = dst_path + ".merged.nc"

    with netCDF4.Dataset(dst_path, "r") as old, \
         netCDF4.Dataset(src_path, "r") as src, \
         netCDF4.Dataset(tmp_merged, "w", format="NETCDF4") as out:

        old_vt  = np.array(old.variables["valid_time"][:])
        old_sst = np.array(old.variables["sst"][:])
        src_vt  = np.array(src.variables["valid_time"][:])
        src_sst = np.array(src.variables["sst"][:])

        old_vt_set = set(int(v) for v in old_vt)
        new_mask = np.array([int(v) not in old_vt_set for v in src_vt])
        n = int(new_mask.sum())
        if n == 0:
            return 0

        all_vt  = np.concatenate([old_vt,  src_vt[new_mask]])
        all_sst = np.concatenate([old_sst, src_sst[new_mask]])
        order   = np.argsort(all_vt)
        all_vt  = all_vt[order]
        all_sst = all_sst[order]

        # измерения
        out.createDimension("valid_time", None)  # UNLIMITED
        out.createDimension("latitude",  len(old.variables["latitude"][:]))
        out.createDimension("longitude", len(old.variables["longitude"][:]))

        for vname in ("latitude", "longitude"):
            v_in = old.variables[vname]
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

        sst_out = out.createVariable("sst", "f4", ("valid_time", "latitude", "longitude"),
                                     zlib=True, complevel=4)
        try:
            sst_out.setncatts({k: old.variables["sst"].getncattr(k)
                               for k in old.variables["sst"].ncattrs()})
        except Exception:
            pass
        sst_out[:] = all_sst

    shutil.move(tmp_merged, dst_path)
    return n


def main():
    parser = argparse.ArgumentParser(description="Докачка ERA5 SST с CDS")
    parser.add_argument("--lag", type=int, default=CDS_LAG_MONTHS,
                        help=f"Задержка CDS в месяцах (по умолчанию {CDS_LAG_MONTHS})")
    parser.add_argument("--run-precompute", action="store_true",
                        help="Запустить precompute.py автоматически после обновления")
    parser.add_argument("--nc", default=NC_FILE,
                        help="Путь к NC файлу (по умолчанию data/sst_era5.nc)")
    args = parser.parse_args()

    nc_path = args.nc

    if not os.path.exists(nc_path):
        print(f"NC файл не найден: {nc_path}")
        print("Скачайте исходные данные ERA5 и положите в data/sst_era5.nc")
        print("Инструкция: см. README раздел 'Данные'")
        sys.exit(1)

    last   = _last_nc_date(nc_path)
    up_y, up_m = _cds_latest() if args.lag == CDS_LAG_MONTHS else (
        date.today().year, max(1, date.today().month - args.lag)
    )

    print(f"Последний шаг в NC:  {last.year}-{last.month:02d}")
    print(f"Доступно на CDS:     {up_y}-{up_m:02d}")

    months = _missing_months(last, up_y, up_m)
    if not months:
        print("Данные актуальны, обновление не требуется")
        return

    print(f"Новых месяцев: {len(months)}")

    tmp = nc_path + ".download.nc"
    try:
        _download(months, tmp)

        if _nc_time_is_unlimited(nc_path):
            added = _append_nc(nc_path, tmp)
        else:
            added = _merge_to_new_file(nc_path, tmp)

        if added > 0:
            last_new = _last_nc_date(nc_path)
            print(f"Добавлено {added} шагов. Последний: {last_new.year}-{last_new.month:02d}")
            print()
            if args.run_precompute:
                import subprocess
                print("Запускаем precompute.py...")
                subprocess.run([sys.executable,
                                os.path.join(os.path.dirname(__file__), "precompute.py")],
                               check=True)
            else:
                print("Данные обновлены. Запустите предвычисление:")
                print("  python precompute.py")
        else:
            print("Новых данных не найдено (возможно, дубликаты)")

    except KeyboardInterrupt:
        print("\nПрервано")
        sys.exit(1)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
