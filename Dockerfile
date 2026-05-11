FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libnetcdf-dev \
    libhdf5-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    gdal-bin \
    libgdal-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

VOLUME ["/app/data"]

EXPOSE 8000

# При первом запуске сервер проверит данные; если AUTO_PREPARE_DATA=1 —
# попытается скачать с CDS (нужен ~/.cdsapirc). Без него — использ
# уже подготовленные .npz.
ENV AUTO_PREPARE_DATA=0

CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]
