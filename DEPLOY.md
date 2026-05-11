# Развёртывание на новой машине

Дашборд состоит из двух частей: **данные** (~700 МБ предвычисленных `.npz`) и
**веб-приложение** на FastAPI/Uvicorn.

## Минимально нужно

- Python 3.10+
- Системные библиотеки: `libnetcdf`, `libgeos`, `libproj`, `libgdal`
- ~2 ГБ диска под `data/`

Если есть готовые `data/*.npz` (см. ниже) — данные не пересчитываются и сервер
поднимается за секунды. Если нет — `prepare_data.py` либо использует уже
лежащие `*.nc` (NetCDF), либо скачивает их с CDS (нужны креды).

## есть готовые data/

Скопируйте папку `data/` целиком на новую машину (любым способом — rsync, S3,
архив). Минимальный набор файлов для запуска:

```
data/
├── meta.npz                         (общие: lat/lon/years)
├── sst_raw.npz, s1.npz … s7.npz     (для ТПО)
├── t2m_meta.npz, t2m_raw.npz,
│   t2m_s1.npz … t2m_s7.npz          (для T2M)
├── ne_10m_land.geojson              (береговая Natural Earth)
└── ocean_mask.npy, ocean_mask_hires.npy  (растровые маски океана)
```

Тогда:

```bash
pip install -r requirements.txt
python prepare_data.py --check       # exit 0 если всё на месте
gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1 --timeout 120
```

## есть только NetCDF

```bash
# data/sst_era5.nc и data/t2m_era5.nc уже лежат
pip install -r requirements.txt
python prepare_data.py --skip-download   # пересчитает все .npz из NC
gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1 --timeout 120
```

Полный пересчёт SST + T2M на ноутбуке занимает ~5–15 минут.

##  пустая машина с нуля (CDS)

Нужен аккаунт на https://cds.climate.copernicus.eu и `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <ваш-uid>:<ваш-key>
```

Затем:

```bash
pip install -r requirements.txt
python prepare_data.py               # скачает SST+T2M (часы), потом пересчитает
```

Качается  NetCDF. Можно отдельно: `--skip-t2m` или `--skip-sst`.

## Docker

```bash
docker build -t sst-dashboard .
docker run -p 8000:8000 -v $(pwd)/data:/app/data sst-dashboard
```

Если данных в `data/` нет — нужен `~/.cdsapirc` + `-e AUTO_PREPARE_DATA=1`:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $HOME/.cdsapirc:/root/.cdsapirc:ro \
  -e AUTO_PREPARE_DATA=1 \
  sst-dashboard
```

## Render.com

`render.yaml` уже настроен. Залейте репозиторий + persistent disk на 5 ГБ.
Данные предварительно положите на disk (через SSH или стейджинг), либо первый
старт скачает с CDS если в env заданы CDS-креды.

## Heroku / другой PaaS

`Procfile` готов:

```
web: gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 120
release: python prepare_data.py --skip-download
```

`release` фаза пересчитает `.npz` из лежащих `.nc` при каждом деплое
(если они новее).

## Обновление данных

Раз в месяц (новый месяц ERA5):

```bash
python prepare_data.py               # скачает новые месяцы и пересчитает
```

## Переменные окружения

| ENV | Эффект |
|---|---|
| `AUTO_PREPARE_DATA=1` | При старте сервера авто-вызвать `prepare_data.ensure_data()` если файлов не хватает |
| `SKIP_DOWNLOAD=1`     | В `AUTO_PREPARE_DATA` пропустить CDS, только локальный пересчёт |
| `SKIP_T2M=1`          | Не требовать `t2m_*.npz` |

## CLI prepare_data.py

```
python prepare_data.py                 # стандартный полный цикл
python prepare_data.py --check         # только проверить наличие; exit 1 если нет
python prepare_data.py --skip-download # без CDS, только пересчёт
python prepare_data.py --skip-t2m      # без t2m
python prepare_data.py --force         # пересчитать NPZ даже если актуальны
python prepare_data.py --lag 3         # CDS-задержка 3 месяца вместо 2
```
