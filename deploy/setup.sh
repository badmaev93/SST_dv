#!/bin/bash
# VPS setup: Ubuntu 20.04 / 22.04 / 24.04
# Запуск: bash setup.sh yourdomain.ru

set -e
DOMAIN=${1:-""}
APP_DIR=/opt/sst-dashboard
APP_USER=sst

echo "==> Обновление системы"
apt-get update -qq && apt-get upgrade -y -qq

echo "==> Базовые пакеты"
apt-get install -y -qq \
    software-properties-common curl git \
    nginx certbot python3-certbot-nginx \
    build-essential libnetcdf-dev libhdf5-dev \
    libgeos-dev libproj-dev proj-data proj-bin \
    ufw fail2ban

# Python 3.11: на Ubuntu 20.04 нужен deadsnakes PPA
if ! command -v python3.11 &>/dev/null; then
    echo "==> Python 3.11 (deadsnakes PPA)"
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-distutils
fi

# pip для системного python (для генерации JWT_SECRET в .env)
apt-get install -y -qq python3-pip 2>/dev/null || true

PY=python3.11

echo "==> Firewall"
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

echo "==> fail2ban"
systemctl enable fail2ban --now

echo "==> Пользователь $APP_USER"
id "$APP_USER" &>/dev/null || useradd -r -s /bin/false -d $APP_DIR $APP_USER

echo "==> Директории"
mkdir -p $APP_DIR/data $APP_DIR/logs
chown -R $APP_USER:$APP_USER $APP_DIR

echo "==> Venv"
$PY -m venv $APP_DIR/venv
$APP_DIR/venv/bin/pip install --upgrade pip -q
$APP_DIR/venv/bin/pip install -r $APP_DIR/requirements.txt -q

echo "==> .env (создаём если нет)"
ENV_FILE=$APP_DIR/.env
if [ ! -f "$ENV_FILE" ]; then
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    cat > "$ENV_FILE" <<ENVEOF
# Обязательно
JWT_SECRET=$JWT_SECRET
AUTH_REQUIRED=1

# Ссылки на приложение (подставляются в mailto при приглашении)
APP_URL_WIN=
APP_URL_MAC=
APP_URL_ANDROID=

# Опционально
JWT_EXP_DAYS=30
SKIP_T2M=1
SST_DATA_DIR=$APP_DIR/data
SST_PORT=8000
ENVEOF
    chmod 600 "$ENV_FILE"
    echo "   .env создан"
fi
chown $APP_USER:$APP_USER "$ENV_FILE"

echo "==> systemd сервис"
cp $APP_DIR/deploy/sst-dashboard.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable sst-dashboard

echo "==> nginx"
cp $APP_DIR/deploy/nginx.conf /etc/nginx/sites-available/sst-dashboard
sed -i "s/__DOMAIN__/${DOMAIN:-_}/g" /etc/nginx/sites-available/sst-dashboard
ln -sf /etc/nginx/sites-available/sst-dashboard /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

if [ -n "$DOMAIN" ]; then
    echo "==> SSL (Let's Encrypt)"
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "admin@$DOMAIN" || \
        echo "   certbot не сработал — настройте SSL вручную"
fi

echo ""
echo "=== Готово ==="
echo "1. Положите .npz файлы в $APP_DIR/data/"
echo "2. Заполните $APP_DIR/.env (SENDGRID_API_KEY)"
echo "3. Создайте первого админа:"
echo "   sudo -u $APP_USER $APP_DIR/venv/bin/python -c \\"
echo "   \"from auth import create_admin; create_admin('admin@yourdomain.ru','yourpassword')\""
echo "4. Запустите сервер: systemctl start sst-dashboard"
echo "5. Проверить: systemctl status sst-dashboard"
