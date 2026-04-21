#!/usr/bin/env python3
"""Предвычисление данных из NetCDF в data/*.npz. Запускать один раз"""
import os, sys, time
import numpy as np
from scipy import stats
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

NC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sst_era5.nc")

# загрузка
print("Loading NetCDF …")
try:
    import netCDF4
    nc  = netCDF4.Dataset(NC_FILE)
    lat = np.array(nc.variables["latitude"][:])
    lon = np.array(nc.variables["longitude"][:])
    vt  = np.array(nc.variables["valid_time"][:])
    sst_raw = np.array(nc.variables["sst"][:], dtype=np.float32)
    nc.close()
except Exception as e:
    print(f"ERROR loading NetCDF: {e}", file=sys.stderr)
    sys.exit(1)

# К в Цельсий, маска NaN
sst = sst_raw - 273.15
sst[np.abs(sst) > 100] = np.nan

NTIME, NLAT, NLON = sst.shape
print(f"  shape: {sst.shape}, lat [{lat[0]:.1f}…{lat[-1]:.1f}], lon [{lon[0]:.1f}…{lon[-1]:.1f}]")

land_mask = np.all(np.isnan(sst), axis=0)   # (NLAT, NLON)

# разбор дат
dates        = [datetime.utcfromtimestamp(int(t)) for t in vt]
years        = np.array([d.year  for d in dates])
months       = np.array([d.month for d in dates])
date_strings = [d.strftime("%Y-%m") for d in dates]

mon_idx = months - 1

# ESS-сезоны: ДЯФ=0, МАМ=1, ИИА=2, СОЯ=3
def ess_season_index(m):
    """0-based month -> 0..3 ДЯФ/МАМ/ИИА/СОЯ"""
    m0 = m - 1
    if m0 in (11, 0, 1):  return 0
    elif m0 in (2, 3, 4): return 1
    elif m0 in (5, 6, 7): return 2
    else:                 return 3

ess_idx = np.array([ess_season_index(m) for m in months])

# OLS-тренд + p-value попиксельно (векторизованно)
def compute_trend(data_nt_nl_nlon, x_years):
    """data: (T, NLAT, NLON) -> тренд C/дес, R2, p-value"""
    T, NL, NW = data_nt_nl_nlon.shape
    trend = np.full((NL, NW), np.nan, dtype=np.float32)
    r2    = np.full((NL, NW), np.nan, dtype=np.float32)
    pval  = np.full((NL, NW), np.nan, dtype=np.float32)
    x  = x_years - x_years.mean()
    sx = np.sum(x); sx2 = np.sum(x**2); n = len(x)
    data_2d = data_nt_nl_nlon.reshape(T, NL * NW)
    valid_cnt = np.sum(np.isfinite(data_2d), axis=0)
    for p in range(NL * NW):
        if valid_cnt[p] < 5:
            continue
        y   = data_2d[:, p]
        msk = np.isfinite(y)
        xm  = x[msk]; ym = y[msk]
        if len(xm) < 5:
            continue
        sl, ic, r, pv, _ = stats.linregress(xm, ym)
        i, j = divmod(p, NW)
        trend[i, j] = sl * 10
        r2[i, j]    = r**2
        pval[i, j]  = pv
    return trend, r2, pval

# перцентильные позиции тиков легенды
TICK_PCTS = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100], dtype=np.float64)

def section_ticks(arrays, land_mask):
    """Пул пикселей из массивов -> тики по перцентилям"""
    chunks = []
    for arr in arrays:
        v = arr[~land_mask & np.isfinite(arr)]
        if v.size > 0:
            # лимит 50k на поле
            step = max(1, len(v) // 50000)
            chunks.append(v[::step])
    if not chunks:
        return np.zeros(len(TICK_PCTS), dtype=np.float32)
    pool = np.concatenate(chunks)
    return np.percentile(pool, TICK_PCTS).astype(np.float32)

# S1: клим. среднее
print("s1: climatological mean …")
s1_monthly = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where(mon_idx == m)[0]
    s1_monthly[m] = np.nanmean(sst[idx], axis=0)

s1_ess = np.full((4, NLAT, NLON), np.nan, dtype=np.float32)
for s in range(4):
    idx = np.where(ess_idx == s)[0]
    s1_ess[s] = np.nanmean(sst[idx], axis=0)

s1_pcts = section_ticks(list(s1_monthly), land_mask)

# S1: клим. среднее по периодам
s1_monthly_1981_2010 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s1_monthly_1991_2020 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s1_ess_1981_2010     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s1_ess_1991_2020     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s1_monthly_1980_2026 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s1_ess_1980_2026     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    i81  = np.where((mon_idx == m) & (years >= 1981) & (years <= 2010))[0]
    i91  = np.where((mon_idx == m) & (years >= 1991) & (years <= 2020))[0]
    i80a = np.where((mon_idx == m) & (years >= 1980))[0]
    if len(i81):  s1_monthly_1981_2010[m] = np.nanmean(sst[i81],  axis=0)
    if len(i91):  s1_monthly_1991_2020[m] = np.nanmean(sst[i91],  axis=0)
    if len(i80a): s1_monthly_1980_2026[m] = np.nanmean(sst[i80a], axis=0)
for s in range(4):
    i81  = np.where((ess_idx == s) & (years >= 1981) & (years <= 2010))[0]
    i91  = np.where((ess_idx == s) & (years >= 1991) & (years <= 2020))[0]
    i80a = np.where((ess_idx == s) & (years >= 1980))[0]
    if len(i81):  s1_ess_1981_2010[s] = np.nanmean(sst[i81],  axis=0)
    if len(i91):  s1_ess_1991_2020[s] = np.nanmean(sst[i91],  axis=0)
    if len(i80a): s1_ess_1980_2026[s] = np.nanmean(sst[i80a], axis=0)

# S2: OLS тренд C/дес (p<0.05)
print("s2: OLS trend …")
x_frac = years + (months - 1) / 12.0

s2_monthly_trend = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s2_monthly_pval  = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where(mon_idx == m)[0]
    if len(idx) < 5: continue
    tr, _, pv = compute_trend(sst[idx], x_frac[idx])
    s2_monthly_trend[m] = np.where(pv < 0.05, tr, np.nan)
    s2_monthly_pval[m]  = pv
    print(f"  month {m+1}/12 done")

s2_ess_trend = np.full((4, NLAT, NLON), np.nan, dtype=np.float32)
for s in range(4):
    idx = np.where(ess_idx == s)[0]
    if len(idx) < 5: continue
    tr, _, pv = compute_trend(sst[idx], x_frac[idx])
    s2_ess_trend[s] = np.where(pv < 0.05, tr, np.nan)
    print(f"  season {s+1}/4 done")

# тики без маски p-value
s2_raw_monthly = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where(mon_idx == m)[0]
    if len(idx) < 5: continue
    tr, _, _ = compute_trend(sst[idx], x_frac[idx])
    s2_raw_monthly[m] = tr
s2_pcts = section_ticks(list(s2_raw_monthly), land_mask)

# S2: тренды по периодам
print("s2: period-specific trends …")
s2_monthly_1981_2010 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s2_monthly_1991_2020 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s2_ess_1981_2010     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s2_ess_1991_2020     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s2_monthly_1980_2026 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s2_ess_1980_2026     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
for period_name, yr0, yr1, arr_m, arr_e in [
    ("1981_2010", 1981, 2010, s2_monthly_1981_2010, s2_ess_1981_2010),
    ("1991_2020", 1991, 2020, s2_monthly_1991_2020, s2_ess_1991_2020),
    ("1980_2026", 1980, 9999, s2_monthly_1980_2026, s2_ess_1980_2026),
]:
    pm = (years >= yr0) & (years <= yr1)
    for m in range(12):
        idx = np.where((mon_idx == m) & pm)[0]
        if len(idx) >= 5:
            tr, _, pv = compute_trend(sst[idx], x_frac[idx])
            arr_m[m] = np.where(pv < 0.05, tr, np.nan)
    for ss in range(4):
        idx = np.where((ess_idx == ss) & pm)[0]
        if len(idx) >= 5:
            tr, _, pv = compute_trend(sst[idx], x_frac[idx])
            arr_e[ss] = np.where(pv < 0.05, tr, np.nan)
    print(f"  period {period_name} done")

# S3: R2
print("s3: R² …")
s3_monthly_r2 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where(mon_idx == m)[0]
    if len(idx) < 5: continue
    _, r2, _ = compute_trend(sst[idx], x_frac[idx])
    s3_monthly_r2[m] = r2

s3_ess_r2 = np.full((4, NLAT, NLON), np.nan, dtype=np.float32)
for s in range(4):
    idx = np.where(ess_idx == s)[0]
    if len(idx) < 5: continue
    _, r2, _ = compute_trend(sst[idx], x_frac[idx])
    s3_ess_r2[s] = r2

s3_pcts = section_ticks(list(s3_monthly_r2), land_mask)

# S3: R2 по периодам
print("s3: period-specific R² …")
s3_monthly_1981_2010 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s3_monthly_1991_2020 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s3_monthly_1980_2026 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s3_ess_1981_2010     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s3_ess_1991_2020     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s3_ess_1980_2026     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
for period_name, yr0, yr1, arr_m, arr_e in [
    ("1981_2010", 1981, 2010, s3_monthly_1981_2010, s3_ess_1981_2010),
    ("1991_2020", 1991, 2020, s3_monthly_1991_2020, s3_ess_1991_2020),
    ("1980_2026", 1980, 9999, s3_monthly_1980_2026, s3_ess_1980_2026),
]:
    pm = (years >= yr0) & (years <= yr1)
    for m in range(12):
        idx = np.where((mon_idx == m) & pm)[0]
        if len(idx) >= 5:
            _, r2, _ = compute_trend(sst[idx], x_frac[idx])
            arr_m[m] = r2
    for ss in range(4):
        idx = np.where((ess_idx == ss) & pm)[0]
        if len(idx) >= 5:
            _, r2, _ = compute_trend(sst[idx], x_frac[idx])
            arr_e[ss] = r2
    print(f"  period {period_name} done")

# S4: межгодовая дисперсия
print("s4: variance …")
s4_monthly = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where(mon_idx == m)[0]
    s4_monthly[m] = np.nanvar(sst[idx], axis=0).astype(np.float32)

s4_ess = np.full((4, NLAT, NLON), np.nan, dtype=np.float32)
for s in range(4):
    idx = np.where(ess_idx == s)[0]
    s4_ess[s] = np.nanvar(sst[idx], axis=0).astype(np.float32)

s4_pcts = section_ticks(list(s4_monthly), land_mask)

# S4: дисперсия по периодам
s4_monthly_1981_2010 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s4_monthly_1991_2020 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s4_monthly_1980_2026 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s4_ess_1981_2010     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s4_ess_1991_2020     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s4_ess_1980_2026     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    i81  = np.where((mon_idx == m) & (years >= 1981) & (years <= 2010))[0]
    i91  = np.where((mon_idx == m) & (years >= 1991) & (years <= 2020))[0]
    i80a = np.where((mon_idx == m) & (years >= 1980))[0]
    if len(i81):  s4_monthly_1981_2010[m] = np.nanvar(sst[i81],  axis=0).astype(np.float32)
    if len(i91):  s4_monthly_1991_2020[m] = np.nanvar(sst[i91],  axis=0).astype(np.float32)
    if len(i80a): s4_monthly_1980_2026[m] = np.nanvar(sst[i80a], axis=0).astype(np.float32)
for ss in range(4):
    i81  = np.where((ess_idx == ss) & (years >= 1981) & (years <= 2010))[0]
    i91  = np.where((ess_idx == ss) & (years >= 1991) & (years <= 2020))[0]
    i80a = np.where((ess_idx == ss) & (years >= 1980))[0]
    if len(i81):  s4_ess_1981_2010[ss] = np.nanvar(sst[i81],  axis=0).astype(np.float32)
    if len(i91):  s4_ess_1991_2020[ss] = np.nanvar(sst[i91],  axis=0).astype(np.float32)
    if len(i80a): s4_ess_1980_2026[ss] = np.nanvar(sst[i80a], axis=0).astype(np.float32)

# S5: тренд дисперсии по квадрату аномалий помесячно (p<0.05)
print("s5: trend of variance …")

def _var_trend_monthly(yr0, yr1):
    result = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
    sel_yrs = np.unique(years[(years >= yr0) & (years <= yr1)])
    if len(sel_yrs) < 5:
        return result
    for m in range(12):
        mon_no = m + 1
        frames, yrs_used = [], []
        for yr in sel_yrs:
            idx = np.where((years == yr) & (months == mon_no))[0]
            if len(idx):
                frames.append(sst[idx[0]])
                yrs_used.append(yr)
        if len(frames) < 5:
            continue
        frames  = np.array(frames, dtype=np.float32)
        y_arr   = np.array(yrs_used, dtype=float)
        clim_m  = np.nanmean(frames, axis=0)
        sq_anom = (frames - clim_m[np.newaxis]) ** 2
        tr, _, pv = compute_trend(sq_anom, y_arr)
        result[m] = np.where(pv < 0.05, tr, np.nan)
    return result

def _var_trend_ess(yr0, yr1):
    result = np.full((4, NLAT, NLON), np.nan, dtype=np.float32)
    sel_yrs = np.unique(years[(years >= yr0) & (years <= yr1)])
    if len(sel_yrs) < 5:
        return result
    for s in range(4):
        frames, yrs_used = [], []
        for yr in sel_yrs:
            idx = np.where((years == yr) & (ess_idx == s))[0]
            if len(idx):
                frames.append(np.nanmean(sst[idx], axis=0))
                yrs_used.append(yr)
        if len(frames) < 5:
            continue
        frames  = np.array(frames, dtype=np.float32)
        y_arr   = np.array(yrs_used, dtype=float)
        clim_s  = np.nanmean(frames, axis=0)
        sq_anom = (frames - clim_s[np.newaxis]) ** 2
        tr, _, pv = compute_trend(sq_anom, y_arr)
        result[s] = np.where(pv < 0.05, tr, np.nan)
    return result

all_yr0 = int(years.min())
all_yr1 = int(years.max())
s5_monthly = _var_trend_monthly(all_yr0, all_yr1)
s5_ess     = _var_trend_ess(all_yr0, all_yr1)

# тики из тренда без маски p
_s5_raw = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    mon_no = m + 1
    frames, yrs_used = [], []
    for yr in np.unique(years):
        idx = np.where((years == yr) & (months == mon_no))[0]
        if len(idx):
            frames.append(sst[idx[0]]); yrs_used.append(yr)
    if len(frames) >= 5:
        frames = np.array(frames, dtype=np.float32)
        y_arr  = np.array(yrs_used, dtype=float)
        clim_m = np.nanmean(frames, axis=0)
        sq_anom = (frames - clim_m[np.newaxis]) ** 2
        tr, _, _ = compute_trend(sq_anom, y_arr)
        _s5_raw[m] = tr
s5_pcts = section_ticks(list(_s5_raw), land_mask)

# S5: по периодам
print("s5: period-specific trend of variance …")
s5_monthly_1981_2010 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s5_monthly_1991_2020 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s5_monthly_1980_2026 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
s5_ess_1981_2010     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s5_ess_1991_2020     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
s5_ess_1980_2026     = np.full((4,  NLAT, NLON), np.nan, dtype=np.float32)
for period_name, yr0, yr1, arr_m, arr_e in [
    ("1981_2010", 1981, 2010, s5_monthly_1981_2010, s5_ess_1981_2010),
    ("1991_2020", 1991, 2020, s5_monthly_1991_2020, s5_ess_1991_2020),
    ("1980_2026", 1980, 9999, s5_monthly_1980_2026, s5_ess_1980_2026),
]:
    actual_yr1 = all_yr1 if yr1 == 9999 else yr1
    arr_m[:] = _var_trend_monthly(yr0, actual_yr1)
    arr_e[:] = _var_trend_ess(yr0, actual_yr1)
    print(f"  period {period_name} done")

# S6: градиент ТПО (фронты) C/100 км
print("s6: gradient / fronts …")
dlat_km  = abs(float(lat[1] - lat[0])) * 111.0
lat_rad  = np.deg2rad(lat)
dlon_deg = abs(float(lon[1] - lon[0]))

def gradient_map(field_2d):
    gy, gx = np.gradient(field_2d, 1.0, 1.0)
    gy_100km = gy / dlat_km * 100.0
    dlon_km  = np.cos(lat_rad)[:, None] * dlon_deg * 111.0
    gx_100km = gx / np.where(dlon_km > 0, dlon_km, 1.0) * 100.0
    return np.sqrt(gx_100km**2 + gy_100km**2).astype(np.float32)

s6_monthly = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    mean = s1_monthly[m].copy()
    mean[land_mask] = np.nan
    g = gradient_map(np.where(np.isnan(mean), 0.0, mean))
    g[land_mask] = np.nan
    s6_monthly[m] = g

s6_ess = np.full((4, NLAT, NLON), np.nan, dtype=np.float32)
for s in range(4):
    mean = s1_ess[s].copy()
    mean[land_mask] = np.nan
    g = gradient_map(np.where(np.isnan(mean), 0.0, mean))
    g[land_mask] = np.nan
    s6_ess[s] = g

s6_pcts = section_ticks(list(s6_monthly), land_mask)

# S7: базовые климатологии для аномалий
print("s7: climatological means for two WMO periods …")

clim_1981_2010 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where((mon_idx == m) & (years >= 1981) & (years <= 2010))[0]
    if len(idx) > 0:
        clim_1981_2010[m] = np.nanmean(sst[idx], axis=0).astype(np.float32)

clim_1991_2020 = np.full((12, NLAT, NLON), np.nan, dtype=np.float32)
for m in range(12):
    idx = np.where((mon_idx == m) & (years >= 1991) & (years <= 2020))[0]
    if len(idx) > 0:
        clim_1991_2020[m] = np.nanmean(sst[idx], axis=0).astype(np.float32)

# пул аномалий для тиков и vm
print("  computing pooled anomalies for ticks/vm …")
anom_chunks = []
for m in range(12):
    idx = np.where(mon_idx == m)[0]
    if len(idx) == 0: continue
    anom = (sst[idx] - clim_1991_2020[m])
    v = anom[:, ~land_mask].ravel()
    v = v[np.isfinite(v)]
    step = max(1, len(v) // 20000)
    anom_chunks.append(v[::step])

all_anom = np.concatenate(anom_chunks) if anom_chunks else np.array([0.0])
anom_abs  = np.abs(all_anom)

anom_vm_99  = float(np.percentile(anom_abs, 99))
anom_vm_999 = float(np.percentile(anom_abs, 99.9))
anom_vm_100 = float(np.max(anom_abs))
print(f"  anom_vm: P99={anom_vm_99:.3f}  P99.9={anom_vm_999:.3f}  P100={anom_vm_100:.3f} C")

s7_pcts = np.percentile(all_anom, TICK_PCTS).astype(np.float32)

# сохранение
print("Saving to data/ …")

np.savez_compressed(os.path.join(DATA_DIR, "meta.npz"),
    lat=lat, lon=lon, date_strings=date_strings,
    years=years, months=months, ess_idx=ess_idx,
    land_mask=land_mask, anom_vm=np.float32(anom_vm_99),
    annual_years=annual_years)

np.savez_compressed(os.path.join(DATA_DIR, "s1.npz"),
    monthly=s1_monthly, ess=s1_ess, pcts=s1_pcts,
    monthly_1981_2010=s1_monthly_1981_2010, ess_1981_2010=s1_ess_1981_2010,
    monthly_1991_2020=s1_monthly_1991_2020, ess_1991_2020=s1_ess_1991_2020,
    monthly_1980_2026=s1_monthly_1980_2026, ess_1980_2026=s1_ess_1980_2026)

np.savez_compressed(os.path.join(DATA_DIR, "s2.npz"),
    monthly=s2_monthly_trend, ess=s2_ess_trend, pcts=s2_pcts,
    monthly_1981_2010=s2_monthly_1981_2010, ess_1981_2010=s2_ess_1981_2010,
    monthly_1991_2020=s2_monthly_1991_2020, ess_1991_2020=s2_ess_1991_2020,
    monthly_1980_2026=s2_monthly_1980_2026, ess_1980_2026=s2_ess_1980_2026)

np.savez_compressed(os.path.join(DATA_DIR, "s3.npz"),
    monthly=s3_monthly_r2, ess=s3_ess_r2, pcts=s3_pcts,
    monthly_1981_2010=s3_monthly_1981_2010, ess_1981_2010=s3_ess_1981_2010,
    monthly_1991_2020=s3_monthly_1991_2020, ess_1991_2020=s3_ess_1991_2020,
    monthly_1980_2026=s3_monthly_1980_2026, ess_1980_2026=s3_ess_1980_2026)

np.savez_compressed(os.path.join(DATA_DIR, "s4.npz"),
    monthly=s4_monthly, ess=s4_ess, pcts=s4_pcts,
    monthly_1981_2010=s4_monthly_1981_2010, ess_1981_2010=s4_ess_1981_2010,
    monthly_1991_2020=s4_monthly_1991_2020, ess_1991_2020=s4_ess_1991_2020,
    monthly_1980_2026=s4_monthly_1980_2026, ess_1980_2026=s4_ess_1980_2026)

np.savez_compressed(os.path.join(DATA_DIR, "s5.npz"),
    monthly=s5_monthly, ess=s5_ess, pcts=s5_pcts,
    monthly_1981_2010=s5_monthly_1981_2010, ess_1981_2010=s5_ess_1981_2010,
    monthly_1991_2020=s5_monthly_1991_2020, ess_1991_2020=s5_ess_1991_2020,
    monthly_1980_2026=s5_monthly_1980_2026, ess_1980_2026=s5_ess_1980_2026)

np.savez_compressed(os.path.join(DATA_DIR, "s6.npz"),
    monthly=s6_monthly, ess=s6_ess, pcts=s6_pcts)

np.savez_compressed(os.path.join(DATA_DIR, "s7.npz"),
    clim_1981_2010=clim_1981_2010,
    clim_1991_2020=clim_1991_2020,
    anom_vm=np.float32(anom_vm_99),
    anom_vm_99=np.float32(anom_vm_99),
    anom_vm_999=np.float32(anom_vm_999),
    anom_vm_100=np.float32(anom_vm_100),
    pcts=s7_pcts)

# исходные ТПО для временных рядов
np.savez_compressed(os.path.join(DATA_DIR, "sst_raw.npz"),
    sst=sst)

print("Done! All data saved to", DATA_DIR)
