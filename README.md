# ТПО Северной Пацифики

Интерактивный дашборд анализа температуры поверхности океана, ERA5 1979-2026.

## Стек

| Компонент | Роль |
|---|---|
| Python / FastAPI / Uvicorn | REST API, раздача статики |
| NumPy / SciPy | численные вычисления, OLS, интерполяция |
| Pillow / netCDF4 / pyproj | рендер PNG, чтение ERA5, проекция Меркатора |
| cdsapi | автоматическое обновление данных с CDS |
| OpenLayers 10.2 | интерактивная карта (EPSG:3857) |
| Chart.js | временные ряды по точкам |
| Vanilla JS | UI без фреймворков |

## Данные

**Исходный файл:** `data/sst_era5.nc` - ERA5 MODA SST (NetCDF4), регион 26-72N 117-246E, переменная `sst` в Кельвинах.

Если у вас уже есть файл ERA5, переименуйте и положите его в `data/sst_era5.nc`.

Для первоначального скачивания (нужна учётная запись CDS, см. ниже):

```python
import cdsapi

request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["sea_surface_temperature"],
    "year": ["1980","1981",...,"2026"],   # все нужные годы
    "month": ["01","02",...,"12"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [72, 117, 26, 246]
}
cdsapi.Client().retrieve("reanalysis-era5-single-levels-monthly-means", request).download("data/sst_era5.nc")
```

## Учётная запись CDS (Copernicus)

Нужна для скачивания ERA5 данных (первоначально и для обновлений).

1. Зарегистрироваться на https://cds.climate.copernicus.eu
2. В личном кабинете скопировать API key
3. Создать файл `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <ваш-api-key>
```

Без этого файла `update_data.py` и первоначальное скачивание работать не будут.

## Быстрый старт

```bash
pip install -r requirements.txt

# положить sst_era5.nc в папку data/
python precompute_coastline.py  # один раз, скачивает маску берегов (~5 с)
python precompute.py            # один раз, ~3-5 мин
uvicorn app:app --port 8080     # запуск
```

Открыть: http://localhost:8080/

## Обновление данных

ERA5 публикует новые месячные данные с задержкой ~2 месяца. Для докачки:

```bash
python update_data.py                   # проверить и скачать новые месяцы
python update_data.py --run-precompute  # скачать + сразу пересчитать
python update_data.py --lag 3           # использовать задержку 3 месяца
```

Скрипт автоматически:
- находит последний месяц в `data/sst_era5.nc`
- сравнивает с доступными данными на CDS
- скачивает только недостающие месяцы
- добавляет их в существующий NC файл

Рекомендуется запускать раз в месяц (например, через cron).

## Структура файлов

```
sst_dashboard_v5/
|- app.py                    # FastAPI сервер, рендер карт, API
|- precompute.py             # предвычисление из NetCDF (один раз)
|- precompute_coastline.py   # маска берегов из Natural Earth (один раз)
|- update_data.py            # докачка новых данных ERA5 с CDS
|- requirements.txt
|- render.yaml         # конфиг для Render.com
|- data/
|  |- sst_era5.nc          # исходные данные ERA5 (~5 ГБ, не в репо)
|  |- ocean_mask.npy      # маска океана 1200*550 (из precompute_coastline.py)
|  |- ne_10m_land.geojson # Natural Earth кэш (не в репо)
|  |- meta.npz        # координаты, даты, land_mask
|  |- sst_raw.npz     # исходные ТПО (float32, ~350 МБ)
|  |- s1.npz          # клим. среднее
|  |- s2.npz          # тренды OLS
|  |- s3.npz          # R2
|  |- s4.npz          # дисперсия
|  |- s5.npz          # тренд дисперсии
|  |- s6.npz          # градиент / фронты
|  |- s7.npz          # климатологии для аномалий
|- static/
   |- index.html
   |- app.js
   |- style.css
   |- vendor/         # OpenLayers 10.2.1, Chart.js (локально)
```

## Деплой

### Render.com

Конфиг `render.yaml` уже в репозитории. Render читает его автоматически при push.

```bash
git push origin main
# buildCommand: pip install -r requirements.txt && python precompute.py
# startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Важно: бесплатный план не включает персистентный диск - данные `.npz` теряются при рестарте. Минимальный платный диск ($1.25/мес) решает проблему. Также бесплатный сервис засыпает после 15 мин простоя.

Обходной путь без платного диска: предвычислить `.npz` локально, загрузить в репозиторий (или S3), убрать `python precompute.py` из buildCommand.

### Railway.app

```toml
# railway.toml
[build]
builder = "nixpacks"
buildCommand = "pip install -r requirements.txt && python precompute.py"

[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port $PORT"
```

Волюм для данных подключается через UI. Бесплатных $5 кредитов хватает примерно на 500 часов работы.

### VPS (Ubuntu)

```bash
git clone <repo> && cd sst_dashboard_v5
pip install -r requirements.txt
python precompute.py
uvicorn app:app --host 0.0.0.0 --port 8080
```

Для продакшена - nginx reverse proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
    }
}
```

Автообновление данных через cron (раз в месяц, 1-го числа в 6:00):

```
0 6 1 * * cd /path/to/sst_dashboard_v5 && python update_data.py --run-precompute >> logs/update.log 2>&1
```

## API

| Endpoint | Описание |
|---|---|
| `GET /api/meta` | метаданные сетки, даты, список секций |
| `GET /api/map/{section}/{season_type}/{season_idx}.png` | карта в PNG (Mercator) |
| `GET /api/colorscale/{section}` | тики и диапазон цветовой шкалы |
| `GET /api/field/{section}/{season_type}/{season_idx}` | сырые данные поля (JSON, Float32) |
| `GET /api/field_value` | значение в точке (lat, lon) |
| `GET /api/point_series` | полный временной ряд точки |
| `GET /api/point_annual` | годовой ряд с выбором месяца/сезона |
| `GET /api/coastline.geojson` | береговая линия (GeoJSON) |
