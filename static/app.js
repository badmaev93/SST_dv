/* SST Dashboard v5 — Frontend */
"use strict";

const MONTHS_RU = [
  "Январь","Февраль","Март","Апрель","Май","Июнь",
  "Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"
];
const MONTHS_SHORT = [
  "Янв","Фев","Мар","Апр","Май","Июн",
  "Июл","Авг","Сен","Окт","Ноя","Дек"
];
const ESS_LABELS_RU = ["Дек–Фев", "Мар–Май", "Июн–Авг", "Сен–Ноя"];

// состояние приложения
const S = {
  section:         "s7",
  avgMode:         "monthly",
  periodIdx:       0,
  yearIdx:         0,
  climPeriod:      "1991_2020",
  percentile:      "99",
  scale:           2,
  smooth:          true,
  blur:            false,
  coastline:       true,
  contours:        true,
  contourLabels:   true,
  _cmpBuildId:     0,
  viewMode:        "compare",   // "compare" | "detail"
  _periodSelected: false,
  meta:            null,
  map:                null,
  imageLayer:         null,
  coastlineLayer:     null,
  popup:              null,
  dbChart:            null,
  dbChart2:           null,
  playTimer:          null,
  _hoverTimer:        null,
  _imageExtent:       null,
  _lastClick:         null,
  // состояние canvas-рендерера
  _fieldCanvasSetup:  false,   // true = baseTile postrender listener attached
  _fieldData:         null,    // decoded field data for client-side rendering
};

// форматирование координат
function formatCoord(lat, lon) {
  const latD = Math.floor(Math.abs(lat)), latM = Math.round((Math.abs(lat)-latD)*60);
  const lonD = Math.floor(Math.abs(lon)), lonM = Math.round((Math.abs(lon)-lonD)*60);
  return `${latD}°${latM}′${lat>=0?"с.ш.":"ю.ш."} ${lonD}°${lonM}′${lon>=0?"в.д.":"з.д."}`;
}

function lonNorm(lon) { return lon > 180 ? lon - 360 : lon; }

function inBounds(lat, lon360) {
  const m = S.meta;
  if (!m) return false;
  const lon = lon360 < 0 ? lon360 + 360 : lon360;
  return lat >= m.lat.min && lat <= m.lat.max && lon >= m.lon.min && lon <= m.lon.max;
}

// инициализация
async function init() {
  showLoading("Загрузка метаданных…");
  try {
    const res = await fetch("/api/meta");
    if (!res.ok) throw new Error(res.statusText);
    S.meta = await res.json();
  } catch (e) {
    hideLoading();
    alert("Не удалось загрузить метаданные.\n" + e);
    return;
  }

  const m = S.meta;
  // берём последний полный год
  const defYear = m.last_complete_year ?? m.years[m.years.length - 1];
  S.yearIdx = m.years.indexOf(defYear);
  if (S.yearIdx < 0) S.yearIdx = m.years.length - 1;

  // Меркатор-экстент от сервера
  S._imageExtent = m.mercator_extent;
  // навигационный экстент (урезан на 5 ячеек ERA5)
  S._dataExtent = m.data_extent || m.mercator_extent;
  // токен рестарта - сброс кэша в compare-режиме
  S._serverTs = m.server_ts || "0";

  try {
    buildYearSlider();
    _populateCmpYearSelect();
    initMap();
    _loadCoastlineLayer();
    syncUI();
    await _applyViewMode();
  } catch (e) {
    console.error("Init error:", e);
    document.getElementById("loading-text").textContent = "Ошибка инициализации: " + e.message;
    return;
  }
  hideLoading();
  // клик вне панели - закрываем
  document.addEventListener("click", _onDocClick);
}

// маска-виньетка вне области данных
function _buildMaskLayer(extent) {
  // 4 прямоугольника вместо полигона-с-дырой (проблемы с winding rule в OL)
  const [xMin, yMin, xMax, yMax] = extent;
  const HUGE = 40075016 * 3;
  const color = "rgba(8,10,22,0.88)";
  const rects = [
    [[-HUGE, yMax],  [HUGE,  yMax],  [HUGE,  HUGE],  [-HUGE, HUGE],  [-HUGE, yMax]],  // top
    [[-HUGE, -HUGE], [HUGE,  -HUGE], [HUGE,  yMin],  [-HUGE, yMin],  [-HUGE, -HUGE]], // bottom
    [[-HUGE, yMin],  [xMin,  yMin],  [xMin,  yMax],  [-HUGE, yMax],  [-HUGE, yMin]],  // left
    [[xMax,  yMin],  [HUGE,  yMin],  [HUGE,  yMax],  [xMax,  yMax],  [xMax,  yMin]],  // right
  ];
  const style = new ol.style.Style({ fill: new ol.style.Fill({ color }) });
  const features = rects.map(coords => {
    const f = new ol.Feature({ geometry: new ol.geom.Polygon([coords]) });
    f.setStyle(style);
    return f;
  });
  return new ol.layer.Vector({
    source: new ol.source.Vector({ features }),
    zIndex: 8,
  });
}

// слой географических подписей
function _buildRuLabelsLayer() {
  // западное полушарие в расширенном Меркаторе: сдвиг на длину экватора
  const EARTH_W = 40075016.686;

  const places = [
    // океаны и моря
    { lon:  175, lat: 40,  text: "Тихий океан",      size: 15, bold: true },
    { lon:  148, lat: 55,  text: "Охотское море",     size: 11, bold: false },
    { lon: -178, lat: 61,  text: "Берингово море",    size: 11, bold: false, ext: true },
    { lon:  134, lat: 38,  text: "Японское море",     size: 10, bold: false },
    // страны
    { lon:  115, lat: 64,  text: "РОССИЯ",            size: 13, bold: true },
    { lon:  150, lat: 66,  text: "РОССИЯ",            size: 12, bold: true },
    { lon: -150, lat: 64,  text: "АЛЯСКА",            size: 12, bold: true, ext: true },
    { lon: -125, lat: 55,  text: "КАНАДА",            size: 13, bold: true, ext: true },
    { lon: -122, lat: 42,  text: "США",               size: 13, bold: true, ext: true },
    { lon:  140, lat: 36,  text: "ЯПОНИЯ",            size: 11, bold: true },
    { lon:  110, lat: 32,  text: "КИТАЙ",             size: 12, bold: true },
    { lon:  128, lat: 37,  text: "КОРЕЯ",             size: 10, bold: false },
  ];

  const features = places.map(p => {
    const coord = ol.proj.fromLonLat([p.lon, p.lat]);
    if (p.ext) coord[0] += EARTH_W;  // shift to extended Mercator
    const f = new ol.Feature({ geometry: new ol.geom.Point(coord) });
    f.setStyle(new ol.style.Style({
      text: new ol.style.Text({
        text:         p.text,
        font:         `${p.bold ? "600" : "400"} ${p.size}px Inter, system-ui`,
        fill:         new ol.style.Fill({ color: "rgba(40,40,60,0.65)" }),
        stroke:       new ol.style.Stroke({ color: "rgba(255,255,255,0.8)", width: 3 }),
        overflow:     true,
        placement:    "point",
        textAlign:    "center",
      }),
    }));
    return f;
  });

  return new ol.layer.Vector({
    source: new ol.source.Vector({ features }),
    zIndex: 15,
    declutter: true,
  });
}

// инициализация карты (EPSG:3857 Меркатор)
function initMap() {
  // расширяем EPSG:3857 для покрытия антимеридиана (117-246 E); без этого OL обрезает ImageStatic на 180 E
  ol.proj.get("EPSG:3857").setExtent(
    [-40075016.686, -20037508.343, 40075016.686, 20037508.343]
  );

  // слой только с изолиниями (прозрачный фон)
  S.imageLayer = new ol.layer.Image({ opacity: 1.0, zIndex: 7, source: null });

  // береговая линия - загружается асинхронно
  S.coastlineLayer = null;

  const graticule = new ol.layer.Graticule({
    strokeStyle: new ol.style.Stroke({
      color: "rgba(120,120,150,0.35)",
      width: 0.6,
      lineDash: [3, 5],
    }),
    showLabels: true,
    wrapX: false,
    zIndex: 20,
    lonLabelStyle: new ol.style.Text({
      font: "10px Inter, monospace",
      fill: new ol.style.Fill({ color: "rgba(60,60,80,0.7)" }),
    }),
    latLabelStyle: new ol.style.Text({
      font: "10px Inter, monospace",
      fill: new ol.style.Fill({ color: "rgba(60,60,80,0.7)" }),
    }),
  });

  const popupEl = document.getElementById("map-popup");
  S.popup = new ol.Overlay({
    element:     popupEl,
    positioning: "bottom-center",
    offset:      [0, -6],
    autoPan:     { animation: { duration: 200 } },
    stopEvent:   true,
  });

  const viewCenter = ol.proj.fromLonLat([160, 60]);
  const initZoom = 4.5;

  const ruLabels = _buildRuLabelsLayer();

  S.map = new ol.Map({
    target:   "map",
    layers:   [S.imageLayer, ruLabels, graticule],
    overlays: [S.popup],
    view: new ol.View({
      projection:          "EPSG:3857",
      center:              viewCenter,
      zoom:                initZoom,
      minZoom:             3,
      maxZoom:             6,
      enableRotation:      false,
      constrainOnlyCenter: false,
      extent:              S._dataExtent,
    }),
  });

  S.map.on("click", onMapClick);

  // клик по попапу - открываем временной ряд
  popupEl.addEventListener("click", () => {
    if (S._lastClick) openTimeseries(S._lastClick);
  });
}

// заполнение выбора года (compare-режим)
function _populateCmpYearSelect() {
  const sel = document.getElementById("cmp-year-select");
  if (!sel || !S.meta) return;
  sel.innerHTML = "";
  S.meta.years.forEach(yr => {
    const opt = document.createElement("option");
    opt.value = yr;
    opt.textContent = yr;
    sel.appendChild(opt);
  });
  sel.value = S.meta.years[S.yearIdx];
}

function setCmpYear(val) {
  const yr = parseInt(val);
  const idx = S.meta.years.indexOf(yr);
  if (idx < 0) return;
  S.yearIdx = idx;
  // синхронизируем слайдер detail-режима
  const slider = document.getElementById("year-slider");
  if (slider) slider.value = idx;
  const yl = document.getElementById("year-label");
  if (yl) yl.textContent = String(yr);
  _buildCompareGrid();
  _refreshLegend();
}

// загрузка векторной береговой линии (GeoJSON)
async function _loadCoastlineLayer() {
  const EARTH_W = 40075016.686;
  const fmt = new ol.format.GeoJSON();
  let data;
  try {
    data = await fetch("/api/coastline.geojson").then(r => r.json());
  } catch (e) {
    console.warn("Coastline load failed:", e);
    return;
  }

  const westFeats = fmt.readFeatures(
    { type: "FeatureCollection", features: data.features.filter(f => f.properties.half === "west") },
    { featureProjection: "EPSG:3857", dataProjection: "EPSG:4326" }
  );
  const eastFeats = fmt.readFeatures(
    { type: "FeatureCollection", features: data.features.filter(f => f.properties.half === "east") },
    { featureProjection: "EPSG:3857", dataProjection: "EPSG:4326" }
  );
  eastFeats.forEach(f => {
    f.getGeometry().applyTransform((coords, out, stride) => {
      stride = stride || 2;
      for (let i = 0; i < coords.length; i += stride) {
        out[i]   = coords[i] + EARTH_W;
        out[i+1] = coords[i+1];
      }
      return out;
    });
  });

  const source = new ol.source.Vector({ features: [...westFeats, ...eastFeats] });

  S.coastlineLayer = new ol.layer.Vector({
    source,
    style: [
      new ol.style.Style({ stroke: new ol.style.Stroke({ color: "rgba(15,25,50,0.12)", width: 3.5 }) }),
      new ol.style.Style({ stroke: new ol.style.Stroke({ color: "rgba(15,25,55,0.50)", width: 0.8 }) }),
    ],
    zIndex: 9,
    visible: S.coastline,
  });
  S.map.addLayer(S.coastlineLayer);
}

// параметры URL карты
function _mapParams(extra = {}) {
  const { section, avgMode, periodIdx, yearIdx, climPeriod, percentile, scale, blur,
          contours, contourLabels, meta } = S;
  const p = {
    percentile,
    clim_period: climPeriod,
    ...(scale > 1 ? { scale } : {}),
    ...(!blur ? { blur: 0 } : {}),
    ...(!contours ? { contours: 0 } : {}),
    ...(contourLabels ? { contour_labels: 1 } : {}),
    ...(section === "s7" ? { year: meta.years[yearIdx] } : {}),
    _v: S._serverTs || "0",
    ...extra,
  };
  return new URLSearchParams(p).toString();
}

// обновление оверлея и легенды
async function _refreshLegend() {
  const { section, avgMode, climPeriod, percentile } = S;
  const seasonType = avgMode === "ess" ? "ess" : "monthly";
  const csUrl = `/api/colorscale/${section}?season_type=${seasonType}&season_idx=0&${_mapParams()}`;
  try {
    const cs = await fetch(csUrl).then(r => r.json());
    if (S.viewMode === "compare") {
      _renderLegendH(cs);
    } else {
      renderLegend(cs);
    }
  } catch (_) {}
}

// горизонтальная полосовая легенда (compare-режим)
function _renderLegendH(cs) {
  const container = document.getElementById("cmp-legend-h");
  if (!container) return;

  const rawUnit = (cs.unit || "").replace(/\/десятилетие/g, "/дек");
  const allTicks = (cs.ticks || []).slice().sort((a, b) => a.norm - b.norm);
  const ticks = allTicks.filter(t => t.norm > 0.02 && t.norm < 0.98);
  const N = ticks.length;
  if (N < 1) { container.innerHTML = ""; return; }

  let canvas = container.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    container.innerHTML = "";
    container.appendChild(canvas);
  }
  const ctx = canvas.getContext("2d");

  const BAND_H  = 28;
  const UNIT_W  = rawUnit ? 38 : 0;
  ctx.font = "bold 10px Inter, system-ui";
  const minBandW = Math.max(34, Math.ceil(
    Math.max(...ticks.map(t => ctx.measureText(t.label).width))
  ) + 10);
  const W = UNIT_W + N * minBandW;
  const H = BAND_H;

  canvas.width  = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  // ячейка единиц слева
  if (rawUnit) {
    ctx.fillStyle = "rgba(230,232,240,0.95)";
    ctx.fillRect(0, 0, UNIT_W, H);
    ctx.font         = "bold 9px Inter, system-ui";
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle    = "#1a2030";
    ctx.fillText(rawUnit, UNIT_W / 2, H / 2);
  }

  // N полос: слева минимум, справа максимум
  ctx.font         = "bold 10px Inter, system-ui";
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";

  ticks.forEach((tick, i) => {
    const x0 = UNIT_W + i * minBandW;
    const lutIdx = Math.max(0, Math.min(255, Math.round(tick.norm * 255)));
    const [r, g, b] = cs.lut[lutIdx];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x0, 0, minBandW, H);

    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    ctx.fillStyle   = lum > 128 ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.95)";
    ctx.shadowColor = lum > 128 ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.5)";
    ctx.shadowBlur  = 2;
    ctx.fillText(tick.label, x0 + minBandW / 2, H / 2);
    ctx.shadowBlur  = 0;
  });
}

// canvas-рендерер на стороне клиента (стиль OL postrender)
// рисует поле прямо в canvas OL через событие 'postrender' базового слоя
// S._fieldCanvasSetup: bool - true когда listener навешан
// S._fieldData: { nlat, nlon, latArr, lonArr, field(Float32Array), landMask(Uint8Array), vmin, vmax, lut }

const _CIRC = 40075016.685578488;
const _HALF = _CIRC / 2;
const _FIELD_OPACITY = 0.85;

function _ensureFieldCanvas() {
  if (S._fieldCanvasSetup) return;
  S._fieldCanvasSetup = true;

  // вешаем на postrender базового слоя - рисуем внутри пайплайна OL
  const baseTile = S.map.getLayers().item(0);
  baseTile.on("postrender", (evt) => {
    if (!S._fieldData) return;
    _paintFieldIntoContext(evt.context, evt.frameState);
  });
}

function _paintFieldIntoContext(ctx, frameState) {
  const d = S._fieldData;
  if (!d) return;

  // физические размеры canvas (OL масштабирует по devicePixelRatio)
  const PR  = frameState.pixelRatio || (window.devicePixelRatio || 1);
  const [W_css, H_css] = frameState.size;
  const PW  = Math.round(W_css * PR);
  const PH  = Math.round(H_css * PR);

  // видимый экстент в метрах EPSG:3857
  const ext = frameState.extent;
  const x0  = ext[0], y0 = ext[1];
  const dw  = ext[2] - x0, dh = ext[3] - y0;

  const { nlat, nlon, latArr, lonArr, field, landMask, vmin, vmax, lut } = d;
  const dv = vmax - vmin || 1;

  // lat ERA5 может убывать (72->26) - обрабатываем корректно
  const latDescending = latArr[0] > latArr[nlat - 1];
  const latMin  = latDescending ? latArr[nlat - 1] : latArr[0];
  const latMax  = latDescending ? latArr[0]        : latArr[nlat - 1];
  const lonMin  = lonArr[0],     lonMax = lonArr[nlon - 1];
  const latStep = (latMax - latMin) / (nlat - 1);
  const lonStep = (lonMax - lonMin) / (nlon - 1);

  // рендер в новый ImageData физического разрешения
  const img = new ImageData(PW, PH);
  const pix = img.data;
  const PI  = Math.PI;

  for (let j = 0; j < PH; j++) {
    // строка j -> Меркатор y (y возрастает вверх)
    const my  = y0 + (PH - 0.5 - j) / PH * dh;
    const lat = Math.atan(Math.sinh(my * PI / _HALF)) * (180 / PI);
    if (lat < latMin - latStep || lat > latMax + latStep) continue;

    // fi_asc: доля (0 = юг, nlat-1 = север)
    const fi_asc = (lat - latMin) / latStep;
    // индекс ERA5 (строка 0 = север когда lat убывает)
    const fi = Math.max(0, Math.min(nlat - 1.0001,
      latDescending ? (nlat - 1) - fi_asc : fi_asc));
    const i0 = Math.floor(fi);
    const t  = fi - i0;

    for (let i = 0; i < PW; i++) {
      const mx  = x0 + (i + 0.5) / PW * dw;
      let   lon = mx * 360.0 / _CIRC;
      // сдвиг в диапазон ERA5 (антимеридиан Тихого океана)
      if (lon < lonMin) lon += 360;
      if (lon > lonMax) lon -= 360;
      if (lon < lonMin || lon > lonMax) continue;

      const fj = Math.max(0, Math.min(nlon - 1.0001, (lon - lonMin) / lonStep));
      const j0 = Math.floor(fj);
      const s  = fj - j0;

      // индексы билинейного стенсиля
      const i00 = i0 * nlon + j0;
      const i10 = i00 + nlon;
      const v00 = field[i00],    v01 = field[i00 + 1];
      const v10 = field[i10],    v11 = field[i10 + 1];
      const l00 = landMask[i00], l01 = landMask[i00 + 1];
      const l10 = landMask[i10], l11 = landMask[i10 + 1];

      const w00 = (1 - t) * (1 - s), w01 = (1 - t) * s;
      const w10 = t       * (1 - s), w11 = t       * s;

      // доля океана в пикселе (1=океан, 0=суша)
      const oceanW = (l00 ? 0 : w00) + (l01 ? 0 : w01) +
                     (l10 ? 0 : w10) + (l11 ? 0 : w11);

      const pidx = (j * PW + i) * 4;

      // суша - серый поверх всего
      if (oceanW < 0.5) {
        pix[pidx]     = 148;
        pix[pidx + 1] = 153;
        pix[pidx + 2] = 160;
        pix[pidx + 3] = 255;
        continue;
      }

      // океан - билинейная интерполяция значения
      let sum = 0, wSum = 0;
      if (!isNaN(v00)) { sum += v00 * w00; wSum += w00; }
      if (!isNaN(v01)) { sum += v01 * w01; wSum += w01; }
      if (!isNaN(v10)) { sum += v10 * w10; wSum += w10; }
      if (!isNaN(v11)) { sum += v11 * w11; wSum += w11; }
      if (wSum < 0.01) continue;  // дыра значимости - прозрачно

      const v    = sum / wSum;
      const norm = Math.max(0, Math.min(1, (v - vmin) / dv));
      const li   = Math.round(norm * 255);
      const clr  = lut[li];
      // мягкий фэдер у берега (oceanW 0.5->1)
      const fadeA = Math.round(Math.min(1, (oceanW - 0.5) / 0.5) * 255 * _FIELD_OPACITY);
      pix[pidx]     = clr[0];
      pix[pidx + 1] = clr[1];
      pix[pidx + 2] = clr[2];
      pix[pidx + 3] = fadeA;
    }
  }

  // композитинг через OffscreenCanvas - прозрачные пиксели не затирают базовую карту
  const off = new OffscreenCanvas(PW, PH);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(off, 0, 0);
  ctx.restore();
}

async function _loadFieldData() {
  const { section, avgMode, periodIdx } = S;
  const seasonType = avgMode === "ess" ? "ess" : "monthly";
  const params = _mapParams();
  const url = `/api/field/${section}/${seasonType}/${periodIdx}?${params}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const json = await resp.json();
    const nlat = json.nlat, nlon = json.nlon;

    function decodeF32(b64) {
      const bin = atob(b64);
      const buf = new ArrayBuffer(bin.length);
      const u8  = new Uint8Array(buf);
      for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
      return new Float32Array(buf);
    }
    function decodeU8(b64) {
      const bin = atob(b64);
      const u8  = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
      return u8;
    }

    const latArr  = decodeF32(json.lat);
    const lonArr  = decodeF32(json.lon);
    const field   = decodeF32(json.field);
    const landMsk = decodeU8(json.land_mask);

    S._fieldData = { nlat, nlon, latArr, lonArr, field, landMask: landMsk,
                     vmin: json.vmin, vmax: json.vmax, lut: json.lut };
    S.map.render();   // принудительный ре-рендер OL -> postrender -> рисуем поле
  } catch (e) {
    console.warn("Field data load failed:", e);
    S._fieldData = null;
    S.map.render();   // убираем устаревший рендер
  }
}

// спиннер загрузки (инжектируется в DOM через JS)
(function _injectSpinnerStyles() {
  const s = document.createElement("style");
  s.textContent = `
    @keyframes _map_spin { to { transform: rotate(360deg); } }
    #_map_spinner_wrap {
      position: fixed; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      z-index: 99999;
      width: 64px; height: 64px; border-radius: 50%;
      background: rgba(255,255,255,0.95);
      box-shadow: 0 4px 20px rgba(0,0,0,0.28);
      display: none; align-items: center; justify-content: center;
    }
    #_map_spinner_wheel {
      width: 36px; height: 36px; border-radius: 50%;
      border: 4px solid rgba(0,80,200,0.15);
      border-top-color: #0055cc;
      animation: _map_spin 0.7s linear infinite;
    }
    #_map_dim_overlay {
      position: fixed; inset: 0; z-index: 99998;
      background: rgba(255,255,255,0.45);
      display: none; pointer-events: none;
    }
  `;
  document.head.appendChild(s);

  const wrap = document.createElement("div");
  wrap.id = "_map_spinner_wrap";
  const wheel = document.createElement("div");
  wheel.id = "_map_spinner_wheel";
  wrap.appendChild(wheel);
  document.body.appendChild(wrap);

  const dim = document.createElement("div");
  dim.id = "_map_dim_overlay";
  document.body.appendChild(dim);
})();

function _showMapLoading() {
  const w = document.getElementById("_map_spinner_wrap");
  const d = document.getElementById("_map_dim_overlay");
  if (w) w.style.display = "flex";
  if (d) d.style.display = "block";
}
function _hideMapLoading() {
  const w = document.getElementById("_map_spinner_wrap");
  const d = document.getElementById("_map_dim_overlay");
  if (w) w.style.display = "none";
  if (d) d.style.display = "none";
}

function refreshMap() {
  const { section, avgMode, periodIdx } = S;
  const seasonType = avgMode === "ess" ? "ess" : "monthly";
  const params = _mapParams();

  S._fieldData = null;

  // спиннер и затемнение сразу
  _showMapLoading();

  // RAF гарантирует отрисовку спиннера до сетевого запроса
  requestAnimationFrame(() => {
    const imageUrl = `/api/map/${section}/${seasonType}/${periodIdx}.png?${params}`;
    const src = new ol.source.ImageStatic({
      url:         imageUrl,
      imageExtent: S._imageExtent,
      crossOrigin: "anonymous",
    });
    const _onDone = () => { _hideMapLoading(); };
    src.on("imageloadend",  _onDone);
    src.on("imageloaderror", _onDone);
    S.imageLayer.setSource(src);
  });

  // цветовая шкала (асинхронно, независимо от загрузки карты)
  const csUrl = `/api/colorscale/${section}?season_type=${seasonType}&season_idx=${periodIdx}&${params}`;
  fetch(csUrl).then(r => r.json()).then(cs => renderLegend(cs)).catch(() => {});
}

// вертикальная легенда: ячейка единиц + N полос
function renderLegend(cs) {
  const rawUnit = (cs.unit || "").replace(/\/десятилетие/g, "/дек");
  // единицы на canvas - скрываем отдельный div
  const unitEl = document.getElementById("legend-unit");
  unitEl.textContent = "";
  unitEl.style.display = "none";

  document.getElementById("legend-ticks").innerHTML = "";

  const canvas = document.getElementById("legend-canvas");
  const ctx    = canvas.getContext("2d");

  const allTicks = (cs.ticks || []).slice().sort((a, b) => a.norm - b.norm);
  const ticks    = allTicks.filter(t => t.norm > 0.02 && t.norm < 0.98);
  const N = ticks.length;

  const UNIT_H  = rawUnit ? 20 : 0;
  const BAND_H  = Math.max(20, Math.floor(280 / Math.max(N, 1)));

  if (N < 1) {
    // фолбэк: простой градиент
    canvas.width  = 52;
    canvas.height = 300;
    ctx.clearRect(0, 0, 52, 300);
    for (let y = 0; y < 300; y++) {
      const idx = Math.round((1 - y / 299) * 255);
      const [r, g, b] = cs.lut[idx];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, y, 52, 1);
    }
    return;
  }

  // измеряем ширину подписей для canvas
  ctx.font = "bold 11px Inter, system-ui";
  const maxLabelW = Math.max(
    ...ticks.map(t => ctx.measureText(t.label).width),
    rawUnit ? ctx.measureText(rawUnit).width : 0
  );
  const PAD = 8;
  const W   = Math.max(44, Math.ceil(maxLabelW) + PAD * 2);
  const H   = UNIT_H + N * BAND_H;

  canvas.width  = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  // ячейка единиц: светлый фон, тёмный текст
  if (rawUnit) {
    ctx.fillStyle = "rgba(240,240,245,0.95)";
    ctx.fillRect(0, 0, W, UNIT_H);
    ctx.font          = "bold 10px Inter, system-ui";
    ctx.textAlign     = "center";
    ctx.textBaseline  = "middle";
    ctx.fillStyle     = "#1a2030";
    ctx.shadowBlur    = 0;
    ctx.fillText(rawUnit, W / 2, UNIT_H / 2);
  }

  // N полос: сверху максимум, снизу минимум
  ctx.font         = "bold 11px Inter, system-ui";
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";

  for (let i = 0; i < N; i++) {
    const tick = ticks[N - 1 - i];   // top→bottom: highest→lowest
    const y0   = UNIT_H + i * BAND_H;

    const lutIdx = Math.max(0, Math.min(255, Math.round(tick.norm * 255)));
    const [r, g, b] = cs.lut[lutIdx];

    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, y0, W, BAND_H);

    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    ctx.fillStyle   = lum > 128 ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.95)";
    ctx.shadowColor = lum > 128 ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.5)";
    ctx.shadowBlur  = 2;
    ctx.fillText(tick.label, W / 2, y0 + BAND_H / 2);
    ctx.shadowBlur  = 0;
  }
}

// синхронизация UI
function syncUI() {
  const { section, avgMode, periodIdx, percentile, climPeriod, scale, smooth, blur } = S;

  document.querySelectorAll(".param-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.section === section);
  });
  document.querySelectorAll(".pct-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.pct === String(percentile));
  });
  document.querySelectorAll(".scale-btn").forEach(b => {
    b.classList.toggle("active", Number(b.dataset.scale) === scale);
  });
  document.querySelectorAll(".smooth-btn").forEach(b => {
    b.classList.toggle("active", (b.dataset.smooth === "1") === smooth);
  });
  document.querySelectorAll(".blur-btn").forEach(b => {
    b.classList.toggle("active", (b.dataset.blur === "1") === blur);
  });
  document.querySelectorAll(".coast-btn").forEach(b => {
    b.classList.toggle("active", (b.dataset.coast === "1") === S.coastline);
  });
  document.querySelectorAll(".contour-btn").forEach(b => {
    b.classList.toggle("active", (b.dataset.contour === "1") === S.contours);
  });
  document.querySelectorAll(".contlbl-btn").forEach(b => {
    b.classList.toggle("active", (b.dataset.contlbl === "1") === S.contourLabels);
  });

  const climSel = document.getElementById("clim-select");
  if (climSel) climSel.value = climPeriod;
  const avgSel = document.getElementById("avg-select");
  if (avgSel) avgSel.value = avgMode;

  buildPeriodButtons();

  // нижняя панель только для s7 в detail-режиме
  const showTimeline = section === "s7" && S.viewMode === "detail";
  document.getElementById("bottom-bar").style.display = showTimeline ? "" : "none";
  if (S.meta && section === "s7") {
    const yl = document.getElementById("year-label");
    if (yl) yl.textContent = String(S.meta.years[S.yearIdx]);
  }

  _updateClimVisibility();
}

function buildPeriodButtons() {
  const { avgMode, periodIdx } = S;
  const labels = avgMode === "ess" ? ESS_LABELS_RU : MONTHS_SHORT;

  document.getElementById("period-label").textContent =
    avgMode === "ess" ? "Сезон" : "Месяц";

  const container = document.getElementById("period-btns");
  container.innerHTML = "";

  labels.forEach((lbl, i) => {
    const btn = document.createElement("button");
    btn.className = "period-btn" + (S._periodSelected && i === periodIdx ? " active" : "");
    btn.textContent = lbl;
    btn.onclick = () => {
      S.periodIdx = i;
      S._periodSelected = true;
      closePanel();
      if (S.viewMode === "compare") {
        S.viewMode = "detail";
        _applyViewMode();
      } else {
        syncUI();
        refreshMap();
      }
    };
    container.appendChild(btn);
  });
}

// слайдер годов
function buildYearSlider() {
  const { years } = S.meta;
  const slider = document.getElementById("year-slider");
  slider.min   = 0;
  slider.max   = years.length - 1;
  slider.value = S.yearIdx;

  const wrap = document.getElementById("year-ticks");
  wrap.innerHTML = "";
  years.forEach((yr, i) => {
    if (yr % 5 !== 0) return;
    const el = document.createElement("span");
    el.className = "yr-tick";
    el.style.left = (i / (years.length - 1) * 100) + "%";
    el.textContent = yr;
    wrap.appendChild(el);
  });
}

function onYearSlider(val) {
  S.yearIdx = parseInt(val);
  const yr = String(S.meta.years[S.yearIdx]);
  const yl = document.getElementById("year-label");
  if (yl) yl.textContent = yr;
  const ys = document.getElementById("cmp-year-select");
  if (ys) ys.value = yr;
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function stepYear(delta) {
  const max = S.meta.years.length - 1;
  S.yearIdx = Math.max(0, Math.min(max, S.yearIdx + delta));
  const yr  = String(S.meta.years[S.yearIdx]);
  const slider = document.getElementById("year-slider");
  if (slider) slider.value = S.yearIdx;
  const yl = document.getElementById("year-label");
  if (yl) yl.textContent = yr;
  const ys = document.getElementById("cmp-year-select");
  if (ys) ys.value = yr;
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function _setPlayBtnText(txt) {
  const el = document.getElementById("play-btn");
  if (el) el.textContent = txt;
}

function togglePlay() {
  if (S.playTimer) {
    clearInterval(S.playTimer);
    S.playTimer = null;
    _setPlayBtnText("▶");
  } else {
    _setPlayBtnText("⏸");
    S.playTimer = setInterval(() => {
      const max = S.meta.years.length - 1;
      if (S.yearIdx >= max) {
        clearInterval(S.playTimer);
        S.playTimer = null;
        _setPlayBtnText("▶");
        return;
      }
      stepYear(1);
    }, 500);
  }
}

// обработчики событий
// сворачиваемые секции левой панели
function toggleLpSection(id) {
  document.getElementById(id).classList.toggle("open");
}

function setAvg(mode) {
  S.avgMode          = mode;
  S.periodIdx        = 0;
  S._periodSelected  = false;
  // синхронизируем avg-select в compare-режиме
  const avgSel = document.getElementById("cmp-avg-select");
  if (avgSel) avgSel.value = mode;
  syncUI();
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function setBlur(on) {
  S.blur = on;
  syncUI();
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function setContours(on) {
  S.contours = on;
  if (!on) S.contourLabels = false;
  syncUI();
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function setContourLabels(on) {
  S.contourLabels = on;
  syncUI();
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function setPct(pct) {
  S.percentile = pct;
  syncUI();
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

function setScale(n) {
  S.scale = n;
  syncUI();
  if (S.viewMode === "compare") _buildCompareGrid();
  else refreshMap();
}

function setSmooth(on) {
  S.smooth = on;
  _applySmoothing();
  syncUI();
  if (S.viewMode === "detail") refreshMap();
  else _buildCompareGrid();
}

// переключение CSS image-rendering
function _applySmoothing() {
  const val = S.smooth ? "auto" : "pixelated";
  document.querySelectorAll("#map canvas, #map .ol-layer canvas").forEach(c => {
    c.style.imageRendering = val;
  });
  document.querySelectorAll(".cmp-img").forEach(img => {
    img.style.imageRendering = val;
  });
  // применяем и после загрузки нового источника
  if (S.imageLayer && S.imageLayer.getSource()) {
    S.map && S.map.once("rendercomplete", () => {
      document.querySelectorAll("#map canvas, #map .ol-layer canvas").forEach(c => {
        c.style.imageRendering = val;
      });
    });
  }
}

// открытие и закрытие панели
function togglePanel() {
  document.getElementById("left-panel").classList.toggle("panel-open");
}
function closePanel() {
  document.getElementById("left-panel").classList.remove("panel-open");
}
function _onDocClick(e) {
  const panel  = document.getElementById("left-panel");
  const toggle = document.getElementById("panel-toggle");
  if (panel.classList.contains("panel-open") &&
      !panel.contains(e.target) &&
      !toggle.contains(e.target)) {
    closePanel();
  }
}


// режим просмотра: compare <-> detail
function toggleMode() {
  S.viewMode = S.viewMode === "compare" ? "detail" : "compare";
  _applyViewMode();
}

function goCompare() {
  S.viewMode = "compare";
  _applyViewMode();
}

function setCmpSection(val) {
  S.section         = val;
  S.periodIdx       = 0;
  S._periodSelected = false;
  _updateClimVisibility();
  // синхронизируем кнопки левой панели
  document.querySelectorAll(".param-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.section === val);
  });
  _syncCmpHeader();
  _buildCompareGrid();
  _refreshLegend();
}

async function _applyViewMode() {
  const compareEl  = document.getElementById("compare-view");
  const backBtn    = document.getElementById("back-btn");
  const mapEl      = document.getElementById("map");
  const legend     = document.getElementById("legend");
  const cmpLegendH = document.getElementById("cmp-legend-h");

  if (S.viewMode === "compare") {
    compareEl.classList.remove("hidden");
    document.getElementById("bottom-bar").style.display = "none";
    if (backBtn)    backBtn.style.display = "none";
    if (mapEl)      mapEl.style.display   = "none";
    if (legend)     legend.classList.add("hidden");
    if (cmpLegendH) cmpLegendH.classList.remove("hidden");
    _syncCmpHeader();
    await _buildCompareGrid();
    await _refreshLegend();
  } else {
    compareEl.classList.add("hidden");
    if (backBtn)    backBtn.style.display = "";
    if (mapEl)      mapEl.style.display   = "";
    if (legend)     legend.classList.remove("hidden");
    if (cmpLegendH) cmpLegendH.classList.add("hidden");
    if (S.map) S.map.updateSize();
    syncUI();
    await refreshMap();
  }
}

function _syncCmpHeader() {
  const ps = document.getElementById("cmp-param-select");
  if (ps) ps.value = S.section;
  const cs = document.getElementById("cmp-clim-select");
  if (cs) cs.value = S.climPeriod;
  const avgSel = document.getElementById("cmp-avg-select");
  if (avgSel) avgSel.value = S.avgMode;
  // выбор года только для s7
  const yw = document.getElementById("cmp-year-wrap");
  if (yw) yw.classList.toggle("hidden", S.section !== "s7");
  const ys = document.getElementById("cmp-year-select");
  if (ys && S.meta) ys.value = S.meta.years[S.yearIdx];
}

// псевдоним для совместимости
function toggleCompare() { toggleMode(); }

async function _buildCompareGrid() {
  // инкремент ID сборки - браузер не подаёт устаревший кэш при смене параметров
  S._cmpBuildId = (S._cmpBuildId || 0) + 1;
  const buildId = S._cmpBuildId;

  const { section, avgMode, climPeriod, percentile, yearIdx, scale, blur, meta } = S;
  const isEss      = avgMode === "ess";
  const labels     = isEss ? ESS_LABELS_RU : MONTHS_RU;
  const seasonType = isEss ? "ess" : "monthly";
  const smooth     = S.smooth ? "auto" : "pixelated";

  const grid = document.getElementById("compare-grid");
  // явно удаляем дочерние элементы перед перестройкой
  while (grid.firstChild) grid.removeChild(grid.firstChild);
  grid.style.gridTemplateColumns = isEss ? "repeat(2, 1fr)" : "repeat(4, 1fr)";

  // миниатюры compare: маленький рендер без изолиний
  const thumbParams = new URLSearchParams({
    percentile,
    clim_period: climPeriod,
    ...(!blur ? { blur: 0 } : {}),
    ...(section === "s7" ? { year: meta.years[yearIdx] } : {}),
    thumb: 1,
    _v: `${S._serverTs || "0"}_${buildId}`,
  }).toString();

  labels.forEach((lbl, i) => {
    const url = `/api/map/${section}/${seasonType}/${i}.png?${thumbParams}`;

    const cell = document.createElement("div");
    cell.className = "cmp-cell" + (S._periodSelected && i === S.periodIdx ? " active" : "");

    const lbl_el = document.createElement("div");
    lbl_el.className = "cmp-label";
    lbl_el.textContent = lbl;

    const wrap = document.createElement("div");
    wrap.className = "cmp-map-wrap";

    const img = document.createElement("img");
    img.className = "cmp-img";
    img.loading   = "eager";
    img.src       = url;
    img.alt       = lbl;
    img.style.imageRendering = smooth;

    const coast = document.createElement("img");
    coast.className = "cmp-coast";
    coast.src       = `/api/coastline.svg?v=${S._serverTs || 1}`;
    coast.alt       = "";
    coast.style.display = S.coastline ? "" : "none";

    wrap.appendChild(img);
    wrap.appendChild(coast);

    cell.onclick = () => {
      S.periodIdx       = i;
      S._periodSelected = true;
      S.viewMode        = "detail";
      closePanel();
      _applyViewMode();
    };

    cell.appendChild(lbl_el);
    cell.appendChild(wrap);
    grid.appendChild(cell);
  });

  const hint = document.getElementById("cmp-hint");
  if (hint) hint.textContent = isEss
    ? "Выберите сезон для подробного просмотра"
    : "Выберите месяц для подробного просмотра";
}

function setClim(clim) {
  S.climPeriod = clim;
  syncUI();
  // синхронизируем оба селекта климатологии
  ["clim-select", "cmp-clim-select"].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = clim;
  });
  if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
  else refreshMap();
}

// переключение береговой линии
function setCoastline(on) {
  S.coastline = on;
  if (S.coastlineLayer) S.coastlineLayer.setVisible(on);
  document.querySelectorAll(".cmp-coast").forEach(el => {
    el.style.display = on ? "" : "none";
  });
  syncUI();
}

function _updateClimVisibility() {
  // 1980-2026 not available for anomalies (s7) — hide option in selects
  ["clim-select", "cmp-clim-select"].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const opt = sel.querySelector('option[value="1980_2026"]');
    if (opt) opt.style.display = S.section === "s7" ? "none" : "";
  });
  // если был 1980-2026 и переключились на s7, сбрасываем на 1991-2020
  if (S.section === "s7" && S.climPeriod === "1980_2026") {
    S.climPeriod = "1991_2020";
    ["clim-select", "cmp-clim-select"].forEach(id => {
      const sel = document.getElementById(id); if (sel) sel.value = "1991_2020";
    });
  }
}

// выпадающий список датасетов
function toggleDatasetMenu() {
  const menu = document.getElementById("dataset-menu");
  menu.classList.toggle("hidden");
  if (!menu.classList.contains("hidden")) {
    const close = (e) => {
      if (!document.getElementById("model-bar").contains(e.target)) {
        menu.classList.add("hidden");
        document.removeEventListener("click", close);
      }
    };
    setTimeout(() => document.addEventListener("click", close), 50);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".param-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      S.section         = btn.dataset.section;
      S.periodIdx       = 0;
      S._periodSelected = false;
      _updateClimVisibility();
      syncUI();
      // синхронизируем селект параметра в compare-шапке
      const ps = document.getElementById("cmp-param-select");
      if (ps) ps.value = S.section;
      if (S.viewMode === "compare") { _buildCompareGrid(); _refreshLegend(); }
      else refreshMap();
    });
  });

  init();
});

// движение указателя - подсказка при наведении
function onPointerMove(evt) {
  if (evt.dragging) return;
  const [lon, lat] = ol.proj.toLonLat(evt.coordinate);
  const tip = document.getElementById("hover-tooltip");

  if (!inBounds(lat, lon)) {
    tip.classList.add("hidden");
    return;
  }

  const px = evt.pixel;
  tip.style.left = px[0] + "px";
  tip.style.top  = px[1] + "px";

  clearTimeout(S._hoverTimer);
  S._hoverTimer = setTimeout(async () => {
    const lon360 = lon < 0 ? lon + 360 : lon;
    const { section, avgMode, periodIdx, yearIdx, climPeriod, meta } = S;
    const seasonType = avgMode === "ess" ? "ess" : "monthly";
    const params = new URLSearchParams({
      section, lat, lon: lon360,
      season_type: seasonType,
      season_idx:  periodIdx,
      clim_period: climPeriod,
      ...(section === "s7" ? { year: meta.years[yearIdx] } : {}),
    });
    try {
      const d = await fetch(`/api/field_value?${params}`).then(r => r.json());
      if (d.value === null) {
        tip.classList.add("hidden");
      } else {
        tip.textContent = `${fmtNum(d.value)} ${d.unit}`;
        tip.classList.remove("hidden");
      }
    } catch (_) {
      tip.classList.add("hidden");
    }
  }, 250);
}

// клик по карте - попап с координатами и значением
async function onMapClick(evt) {
  const [lon, lat] = ol.proj.toLonLat(evt.coordinate);
  if (!inBounds(lat, lon)) {
    _hidePopup();
    return;
  }

  const lon360 = lon < 0 ? lon + 360 : lon;
  const { section, avgMode, periodIdx, yearIdx, climPeriod, meta } = S;
  const seasonType = avgMode === "ess" ? "ess" : "monthly";

  // значение для текущей секции
  const params = new URLSearchParams({
    section, lat, lon: lon360,
    season_type: seasonType,
    season_idx:  periodIdx,
    clim_period: climPeriod,
    ...(section === "s7" ? { year: meta.years[yearIdx] } : {}),
  });

  let value = null, unit = "";
  try {
    const d = await fetch(`/api/field_value?${params}`).then(r => r.json());
    value = d.value;
    unit  = d.unit || "";
  } catch (_) {}

  if (value === null) { _hidePopup(); return; }

  const coordStr = formatCoord(lat, lonNorm(lon360));
  const valStr   = `${fmtNum(value)} ${unit}`.trim();

  // сохраняем для перехода попап -> временной ряд
  S._lastClick = { lat, lon360, coord: evt.coordinate, value, unit, coordStr };

  // компактный попап: иконка-булавка, координаты, значение
  const pinSvg = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" style="vertical-align:middle;margin-right:4px"><circle cx="12" cy="10" r="3"/><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/></svg>`;
  const popupEl = document.getElementById("map-popup");
  document.getElementById("map-popup-content").innerHTML =
    `<div class="popup-coord">${pinSvg}${coordStr} &nbsp;|&nbsp; <b>${valStr}</b></div>`;
  popupEl.style.display = "";
  S.popup.setPosition(evt.coordinate);
}

function _hidePopup() {
  document.getElementById("map-popup").style.display = "none";
  S.popup.setPosition(undefined);
  S._lastClick = null;
}

// вспомогательные функции для графиков
function _runningMean(values, w) {
  return values.map((_, i) => {
    const half = Math.floor(w / 2);
    const s = Math.max(0, i - half), e = Math.min(values.length, i + half + 1);
    const chunk = values.slice(s, e).filter(v => v !== null && v !== undefined);
    return chunk.length > 0 ? chunk.reduce((a, b) => a + b, 0) / chunk.length : null;
  });
}

function _linearTrend(values) {
  const pts = values.map((v, i) => [i, v]).filter(p => p[1] !== null && p[1] !== undefined);
  if (pts.length < 3) return values.map(() => null);
  const n = pts.length;
  const mx = pts.reduce((a, p) => a + p[0], 0) / n;
  const my = pts.reduce((a, p) => a + p[1], 0) / n;
  const ss = pts.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0);
  const sx = pts.reduce((a, p) => a + (p[0] - mx) ** 2, 0);
  if (sx === 0) return values.map(() => my);
  const slope = ss / sx, intercept = my - slope * mx;
  return values.map((_, i) => Math.round((slope * i + intercept) * 1000) / 1000);
}

function _annualMean(labels, values) {
  const byYear = {};
  labels.forEach((lbl, i) => {
    const yr = lbl.slice(0, 4);
    if (!byYear[yr]) byYear[yr] = [];
    if (values[i] !== null) byYear[yr].push(values[i]);
  });
  const yrs = Object.keys(byYear).sort();
  return {
    labels: yrs,
    values: yrs.map(y => {
      const arr = byYear[y];
      return arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length * 1000) / 1000 : null;
    }),
  };
}

// число с запятой как разделителем (русский формат)
function fmtNum(v, dec = 2) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return v.toFixed(dec).replace(".", ",");
}

const _chartDefaults = {
  responsive: true, maintainAspectRatio: false, animation: false,
  interaction: { mode: "index", intersect: false },
  plugins: {
    legend: { labels: { color: "rgba(30,40,70,0.65)", boxWidth: 14, font: { size: 10 }, padding: 8 } },
    tooltip: {
      backgroundColor: "rgba(255,255,255,0.97)",
      borderColor: "rgba(0,80,160,0.2)", borderWidth: 1,
      titleColor: "rgba(30,40,70,0.65)", bodyColor: "#1a2030",
      callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y != null ? fmtNum(ctx.parsed.y) : "—"}` },
    },
  },
  scales: {
    x: {
      ticks: { color: "rgba(30,40,70,0.45)", maxTicksLimit: 10, maxRotation: 0, font: { size: 9 } },
      grid:  { color: "rgba(0,0,0,0.06)" },
    },
    y: {
      ticks: { color: "rgba(30,40,70,0.45)", font: { size: 9 },
               callback: v => fmtNum(v) },
      grid:  { color: "rgba(0,0,0,0.07)" },
    },
  },
};

// панель временного ряда (выезжает справа)
async function openTimeseries(clickData) {
  _hidePopup();

  const { lat, lon360, coordStr } = clickData;
  const { section, climPeriod, avgMode, periodIdx } = S;

  document.getElementById("db-title").textContent = coordStr;
  document.getElementById("dashboard").classList.remove("hidden");

  // уничтожаем старые графики
  if (S.dbChart)   { S.dbChart.destroy();   S.dbChart   = null; }
  if (S.dbChart2)  { S.dbChart2.destroy();  S.dbChart2  = null; }

  const dbBody = document.getElementById("db-body");
  dbBody.innerHTML = "";

  try {
    const params = new URLSearchParams({ section, lat, lon: lon360, clim: climPeriod });
    const res = await fetch(`/api/point_series?${params}`);
    if (!res.ok) throw new Error(res.statusText);
    const d = await res.json();

    const sectionInfo = {
      s1: { title: "Клим. среднее ТПО",     unit: "°C",              type: "seasonal" },
      s2: { title: "Тренд ТПО (p<0.05)",    unit: "°C/дек",  type: "bar_trend" },
      s3: { title: "Детерминация тренда R²", unit: "",       type: "bar_r2" },
      s4: { title: "Дисперсия ТПО",         unit: "°C²",    type: "variance_trend_only" },
      s5: { title: "Тренд дисперсии ТПО",   unit: "°C²/дек", type: "bar_trend" },
      s7: { title: "Аномалия ТПО",          unit: "°C",              type: "anomaly" },
    };
    const info = sectionInfo[section] || { title: section, unit: "", type: "line" };

    // строка статистики
    const statsDiv = _el("div", "db-stats");
    const valid = d.values.filter(v => v !== null);
    if (valid.length) {
      const mn  = Math.min(...valid);
      const mx  = Math.max(...valid);
      const avg = valid.reduce((a, b) => a + b, 0) / valid.length;
      statsDiv.innerHTML =
        `<div class="db-stat-row"><span>Параметр</span><span class="db-stat-val">${info.title}</span></div>` +
        `<div class="db-stat-row"><span>Мин / Макс</span><span class="db-stat-val">${fmtNum(mn)} / ${fmtNum(mx)} ${d.unit}</span></div>` +
        `<div class="db-stat-row"><span>Среднее</span><span class="db-stat-val">${fmtNum(avg)} ${d.unit}</span></div>`;
    }
    dbBody.appendChild(statsDiv);

    // основной график (для s1 и s7 включает вспомогательные селекторы)
    _buildMainChart(dbBody, d, info, section, clickData);

  } catch (e) {
    dbBody.innerHTML = `<div class="db-stats"><div class="db-stat-row"><span style="color:#f87171">Ошибка загрузки данных</span></div></div>`;
  }
}

function _el(tag, cls, html = "") {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

function _chartWrap(parent, heightPx = 180) {
  const wrap = _el("div", "db-chart-wrap");
  wrap.style.height = heightPx + "px";
  const cv = document.createElement("canvas");
  wrap.appendChild(cv);
  parent.appendChild(wrap);
  return cv;
}

// вспомогательные функции графиков
function _anomalyBarChart(ctx, labels, values, unit, chartRef) {
  const trend = _linearTrend(values);
  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: `Аномалия (${unit})`,
          data:  values,
          backgroundColor: values.map(v => v === null ? "transparent"
            : v > 0 ? "rgba(248,113,113,0.78)" : "rgba(79,195,247,0.78)"),
          borderWidth: 0, type: "bar",
        },
        {
          label: "Тренд",
          data:  trend,
          borderColor: "rgba(240,192,64,0.95)", backgroundColor: "transparent",
          borderWidth: 2.5, pointRadius: 0, fill: false, tension: 0, spanGaps: true,
          type: "line",
        },
      ],
    },
    options: { ..._chartDefaults,
      scales: { ..._chartDefaults.scales,
        y: { ..._chartDefaults.scales.y,
          title: { display: true, text: unit, color: "rgba(30,40,70,0.4)", font: { size: 9 } } },
      },
    },
  });
  return chart;
}

function _buildMainChart(parent, d, info, section, clickData) {
  const titleDiv = _el("div", "db-section-title", info.title);
  parent.appendChild(titleDiv);

  // столбчатые графики (месячный тренд / R2)
  if (info.type === "bar_trend" || info.type === "bar_r2") {
    const cv  = _chartWrap(parent, 180);
    const ctx = cv.getContext("2d");
    const colors = d.values.map(v =>
      v === null ? "transparent" :
      info.type === "bar_trend"
        ? (v > 0 ? "rgba(248,113,113,0.82)" : "rgba(79,195,247,0.82)")
        : "rgba(250,200,50,0.82)"
    );
    S.dbChart = new Chart(ctx, {
      type: "bar",
      data: { labels: d.labels,
        datasets: [{ label: `${info.title} (${d.unit})`, data: d.values,
          backgroundColor: colors, borderWidth: 0 }] },
      options: { ..._chartDefaults,
        scales: { ..._chartDefaults.scales,
          y: { ..._chartDefaults.scales.y,
            title: { display: !!d.unit, text: d.unit, color: "rgba(30,40,70,0.4)", font: { size: 9 } } } } },
    });
    return;
  }

  // s1: сезонный цикл + годовой ряд
  if (info.type === "seasonal") {
    const cv  = _chartWrap(parent, 160);
    const ctx = cv.getContext("2d");
    S.dbChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: d.labels,
        datasets: [{
          label: `Клим. среднее (${d.unit})`,
          data:  d.values,
          borderColor: "#4fc3f7", backgroundColor: "rgba(79,195,247,0.13)",
          borderWidth: 2.5, pointRadius: 4, pointBackgroundColor: "#4fc3f7",
          fill: true, tension: 0.4, spanGaps: true,
        }],
      },
      options: _chartDefaults,
    });
    // виджет годового ряда
    _buildAnnualWidget(parent, section, clickData, "s1");
    return;
  }

  // s4: годовой ряд дисперсии
  if (info.type === "variance_trend_only") {
    const cv  = _chartWrap(parent, 180);
    const ctx = cv.getContext("2d");
    const trend = _linearTrend(d.values);
    const rm    = _runningMean(d.values, 5);
    S.dbChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: d.labels,
        datasets: [
          { label: `Дисперсия (${d.unit})`, data: d.values,
            borderColor: "rgba(79,195,247,0.55)", backgroundColor: "rgba(79,195,247,0.09)",
            borderWidth: 1.2, pointRadius: 0, fill: true, tension: 0.2, spanGaps: true },
          { label: "Скол. ср. 5 лет", data: rm,
            borderColor: "#f0c040", borderWidth: 2.5, pointRadius: 0, fill: false, spanGaps: true },
          { label: "Тренд", data: trend,
            borderColor: "rgba(248,113,113,0.75)", borderWidth: 1.5, borderDash: [4, 3],
            pointRadius: 0, fill: false, spanGaps: true },
        ],
      },
      options: { ..._chartDefaults,
        scales: { ..._chartDefaults.scales,
          y: { ..._chartDefaults.scales.y,
            title: { display: true, text: d.unit, color: "rgba(30,40,70,0.4)", font: { size: 9 } } } } },
    });
    return;
  }

  // s7: столбчатая аномалия с селектором
  if (info.type === "anomaly") {
    _buildAnnualWidget(parent, section, clickData, "s7");
    return;
  }
}

// виджет годового ряда/аномалий с выбором месяца или сезона
function _buildAnnualWidget(parent, section, clickData, _sec) {
  const { lat, lon360 } = clickData;
  const { climPeriod }  = S;

  const MONTHS_BTN = ["Все",
    "Янв","Фев","Мар","Апр","Май","Июн",
    "Июл","Авг","Сен","Окт","Ноя","Дек"];
  const ESS_BTN = ["Д–Ф","М–М","И–А","С–Н"];

  const titleLabel = _sec === "s7" ? "Годовые аномалии" : "Среднее по годам";
  parent.appendChild(_el("div", "db-section-title", titleLabel));

  const selectorDiv = _el("div", "db-annual-selector");
  const allBtns = [];
  const makeBtn = (label, isActive, onClick) => {
    const b = _el("button", "db-mo-btn" + (isActive ? " active" : ""));
    b.textContent = label;
    b.onclick = () => { allBtns.forEach(x => x.classList.remove("active")); b.classList.add("active"); onClick(); };
    allBtns.push(b);
    selectorDiv.appendChild(b);
    return b;
  };
  parent.appendChild(selectorDiv);

  const chartWrapDiv = _el("div", "db-chart-wrap");
  chartWrapDiv.style.height = "155px";
  const chartCv = document.createElement("canvas");
  chartWrapDiv.appendChild(chartCv);
  parent.appendChild(chartWrapDiv);
  const chartCtx = chartCv.getContext("2d");
  let activeChart = null;

  const _refresh = async (month, ess) => {
    if (activeChart) { activeChart.destroy(); activeChart = null; }
    const params = new URLSearchParams({ section: _sec, lat, lon: lon360, clim: climPeriod });
    if (ess   != null) params.set("ess",   ess);
    else if (month != null) params.set("month", month);
    try {
      const ann = await fetch(`/api/point_annual?${params}`).then(r => r.json());
      if (_sec === "s7") {
        activeChart = _anomalyBarChart(chartCtx, ann.labels, ann.values, ann.unit, "active");
        S.dbChart2  = activeChart;
      } else {
        const trend = _linearTrend(ann.values);
        const rm    = _runningMean(ann.values, 7);
        activeChart = new Chart(chartCtx, {
          type: "line",
          data: { labels: ann.labels, datasets: [
            { label: "Среднее ТПО (°C)", data: ann.values,
              borderColor: "#4fc3f7", backgroundColor: "rgba(79,195,247,0.1)",
              borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.2, spanGaps: true },
            { label: "Скол. ср. 7 лет", data: rm,
              borderColor: "#f0c040", borderWidth: 2.5, pointRadius: 0, fill: false, spanGaps: true },
            { label: "Тренд", data: trend,
              borderColor: "rgba(248,113,113,0.8)", borderDash: [4, 3],
              borderWidth: 1.5, pointRadius: 0, fill: false, spanGaps: true },
          ]},
          options: _chartDefaults,
        });
        S.dbChart2 = activeChart;
      }
    } catch (_) {}
  };

  makeBtn("Все", true, () => _refresh(null, null));
  MONTHS_BTN.slice(1).forEach((lbl, mi) => makeBtn(lbl, false, () => _refresh(mi, null)));
  ESS_BTN.forEach((lbl, si)             => makeBtn(lbl, false, () => _refresh(null, si)));
  _refresh(null, null);
}

function _buildAuxChart() {} // noop — logic merged into _buildMainChart


function closeDashboard() {
  document.getElementById("dashboard").classList.add("hidden");
  if (S.dbChart)  { S.dbChart.destroy();  S.dbChart  = null; }
  if (S.dbChart2) { S.dbChart2.destroy(); S.dbChart2 = null; }
}

// оверлей загрузки
function showLoading(text) {
  document.getElementById("loading-text").textContent = text || "Загрузка…";
  document.getElementById("loading-overlay").classList.remove("hidden");
}

function hideLoading() {
  document.getElementById("loading-overlay").classList.add("hidden");
}
