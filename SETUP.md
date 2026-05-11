# ERA5 ДВ — Инструкция по развёртыванию

## Архитектура

```
VPS (Flask/FastAPI) ←──── данные .npz в /opt/sst-dashboard/data/
       ↑
       │  HTTPS + JWT
       ├── Electron (Windows / macOS)
       └── Android (WebView / RuStore)
```

Все клиенты (десктоп и мобильный) обращаются к одному серверу на VPS.  
Авторизация — invite-only: пользователи добавляются только по приглашению через email.

---

## 1. Сервер (VPS)

### Требования
- Ubuntu 22.04+, от 2 ГБ RAM
- Домен (для SSL), например `sst.yourdomain.ru`

### Установка

```bash
# Загрузить код на VPS
scp -r . root@YOUR_VPS_IP:/opt/sst-dashboard/

# Подключиться и запустить setup
ssh root@YOUR_VPS_IP
mkdir -p /opt/sst-dashboard   # если не создана
bash /opt/sst-dashboard/deploy/setup.sh sst.yourdomain.ru
```

Скрипт установит: Python 3.11, nginx, SSL, fail2ban, systemd-сервис.

### Настройка .env

После установки отредактировать `/opt/sst-dashboard/.env`:

```env
AUTH_REQUIRED=1
JWT_SECRET=<генерируется автоматически, не менять>
APP_URL_WIN=                           # GitHub Releases — заполнить когда будет готово
APP_URL_MAC=                           # GitHub Releases — заполнить когда будет готово
APP_URL_ANDROID=                       # RuStore — заполнить когда будет готово
JWT_EXP_DAYS=30
SKIP_T2M=1
SST_DATA_DIR=/opt/sst-dashboard/data
```

### Загрузить данные

```bash
# Скопировать предвычисленные .npz файлы
scp data/*.npz root@YOUR_VPS_IP:/opt/sst-dashboard/data/
scp data/ocean_mask*.npy root@YOUR_VPS_IP:/opt/sst-dashboard/data/
```

### Создать первого администратора

```bash
ssh root@YOUR_VPS_IP
cd /opt/sst-dashboard
venv/bin/python -c "from auth import create_admin; create_admin('admin@yourdomain.ru', 'ваш_пароль')"
```

### Запуск

```bash
systemctl start sst-dashboard
systemctl status sst-dashboard   # проверить что запустилось
```

Логи: `/opt/sst-dashboard/logs/error.log`

---

## 2. Управление пользователями

Через API (авторизоваться как админ):

```bash
# Получить токен
TOKEN=$(curl -s -X POST https://sst.yourdomain.ru/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@yourdomain.ru","password":"пароль"}' | jq -r .token)

# Пригласить нового пользователя — в ответе придёт mailto: ссылка
# Открыть её в браузере — откроется почтовый клиент с готовым письмом
curl -X POST https://sst.yourdomain.ru/auth/invite \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com"}' | jq -r .mailto

# Список пользователей
curl https://sst.yourdomain.ru/auth/users \
  -H "Authorization: Bearer $TOKEN"

# Деактивировать пользователя
curl -X POST "https://sst.yourdomain.ru/auth/deactivate?email=user@example.com" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 3. Десктоп (Windows / macOS)

### Требования
- Node.js 18+
- npm

### Сборка

```bash
cd electron
npm install

# macOS (DMG, Intel + Apple Silicon)
npm run build:mac

# Windows (NSIS installer)
npm run build:win
```

Готовые файлы в `electron/dist/`.

### Первый запуск у пользователя

При первом запуске приложение спросит адрес сервера — ввести `https://sst.yourdomain.ru`.  
Адрес сохраняется, при следующих запусках не спрашивается.

Для сборки с вшитым URL (чтобы не спрашивал):
```bash
SST_SERVER_URL=https://sst.yourdomain.ru npm run build:mac
```

---

## 4. Android (RuStore / Google Play)

> Раздел в разработке — потребуется Capacitor или WebView-враппер.

Текущий статус: веб-интерфейс полностью адаптирован для мобильных браузеров.  
Временное решение: пользователи открывают `https://sst.yourdomain.ru` в браузере и добавляют в «На главный экран» (PWA).

Для публикации в **RuStore**:
1. Зарегистрироваться на rustore.ru/developer (через Госуслуги)
2. Собрать APK через Capacitor (инструкция будет добавлена)

---

## Структура проекта

```
app.py              — FastAPI сервер (API + рендер карт)
auth.py             — авторизация (JWT + SQLite, приглашения через mailto)
precompute.py       — предвычисление .npz из .nc файлов
prepare_data.py     — скачивание данных ERA5 с CDS
static/             — фронтенд (OpenLayers, Chart.js)
electron/           — обёртка для Windows/macOS
deploy/             — setup.sh, nginx.conf, systemd сервис
data/               — данные (не в git, передавать отдельно)
requirements.txt    — Python зависимости
```

## Частые проблемы

| Проблема | Решение |
|---|---|
| `502 Bad Gateway` | `systemctl restart sst-dashboard` |
| Данные не загружаются | Проверить что `.npz` файлы в `/opt/sst-dashboard/data/` |
| mailto не открывается | Убедиться что в браузере/ОС есть почтовый клиент по умолчанию |
| SSL не работает | `certbot renew --dry-run`, проверить DNS |
| Токен истёк | Пользователь должен перелогиниться, или увеличить `JWT_EXP_DAYS` |
