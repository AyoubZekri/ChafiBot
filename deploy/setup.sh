#!/usr/bin/env bash
# تثبيت ChafiBot على VPS (Ubuntu 22.04 / 24.04) — مرة واحدة.
#
# الاستعمال (كـ root، من مجلد المشروع):
#   DOMAIN=ai.chafi.net EMAIL=you@example.com bash deploy/setup.sh
#
# - DOMAIN: نطاق فرعي يشير إلى IP هذا الـ VPS (لـ HTTPS). بدونه: http://IP (للتجربة فقط).
# - EMAIL: لتنبيهات شهادة HTTPS (اختياري).
# - LLM_MODEL: لفرض نموذج معين (افتراضياً حسب ذاكرة الـ VPS).
#
# آمن للإعادة: كل خطوة تتحقق إن كانت منجزة.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER=chafibot
DOMAIN="${DOMAIN:-}"
EMAIL="${EMAIL:-}"

[ "$(id -u)" -eq 0 ] || { echo "❌ شغّل السكريبت كـ root"; exit 1; }
cd "$APP_DIR"

echo "==> 1/9 الحزم"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-pip git curl nginx ufw zstd

echo "==> 2/9 Swap 4GB (حماية من نفاد الذاكرة)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> 3/9 Ollama (يستمع محلياً فقط، والنموذج يبقى في الذاكرة)"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done

echo "==> 4/9 النموذج حسب ذاكرة الـ VPS"
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024 + 0.5}' /proc/meminfo)
if [ -z "${LLM_MODEL:-}" ]; then
  if [ "$MEM_GB" -ge 15 ]; then LLM_MODEL="qwen2.5:7b"; else LLM_MODEL="qwen3:4b"; fi
fi
echo "    الذاكرة: ${MEM_GB}GB | الأنوية: $(nproc) | النموذج: $LLM_MODEL"
ollama pull "$LLM_MODEL"

echo "==> 5/9 مستخدم الخدمة (لا تعمل كـ root)"
id -u "$APP_USER" >/dev/null 2>&1 || \
  useradd --system --create-home --home-dir /var/lib/$APP_USER --shell /usr/sbin/nologin "$APP_USER"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
git config --global --add safe.directory "$APP_DIR" || true

echo "==> 6/9 بيئة Python (PyTorch للمعالج: أخف بكثير من نسخة GPU)"
sudo -H -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
PIP="sudo -H -u $APP_USER $APP_DIR/.venv/bin/pip"
$PIP install --upgrade pip
$PIP install torch --index-url https://download.pytorch.org/whl/cpu
$PIP install -r "$APP_DIR/requirements.txt"

echo "==> 7/9 الإعدادات (.env)"
if [ ! -f "$APP_DIR/.env" ]; then
  KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  cat > "$APP_DIR/.env" <<EOF
# قاعدة بيانات Chafi في Hostinger (أضف IP هذا الـ VPS في hPanel ← Remote MySQL)
DB_HOST=chafi.net
DB_PORT=3306
DB_DATABASE=
DB_USERNAME=
DB_PASSWORD=

EMBED_MODEL=intfloat/multilingual-e5-small
# نسخة خفيفة من نموذج إعادة الترتيب: مناسبة للمعالج بدون GPU
RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
OLLAMA_URL=http://127.0.0.1:11434
LLM_MODEL=$LLM_MODEL

NUM_PREDICT=350
NUM_CTX=4096
NUM_THREAD=$(nproc)
KEEP_ALIVE=-1
DISABLE_THINKING=1
TOP_K=5
MAX_REWRITES=1
MAX_TOOL_ROUNDS=2
HISTORY_LIMIT=6

# نفس القيمة في .env الخاص بـ Laravel (AI_SERVICE_KEY)
AI_SERVICE_KEY=$KEY
EOF
  chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
fi

echo "==> 8/9 الخدمات (تعمل وحدها عند إعادة تشغيل الـ VPS)"
cat > /etc/systemd/system/chafibot.service <<EOF
[Unit]
Description=ChafiBot (FastAPI + LangGraph)
After=network-online.target ollama.service
Wants=network-online.target ollama.service

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/uvicorn service:app --host 127.0.0.1 --port 8001 --workers 1
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# إعادة بناء الفهرس كل ليلة (حتى تظهر المواد الجديدة من لوحة التحكم)
cat > /etc/systemd/system/chafibot-index.service <<EOF
[Unit]
Description=ChafiBot: إعادة بناء الفهرس
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/bash $APP_DIR/deploy/rebuild-index.sh
EOF
cat > /etc/systemd/system/chafibot-index.timer <<'EOF'
[Unit]
Description=ChafiBot: إعادة بناء الفهرس كل ليلة

[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true

[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable chafibot chafibot-index.timer
systemctl start chafibot-index.timer

echo "==> 9/9 Nginx + جدار الحماية"
SERVER_NAME="${DOMAIN:-_}"
cat > /etc/nginx/sites-available/chafibot <<EOF
server {
    listen 80;
    server_name $SERVER_NAME;
    client_max_body_size 1m;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        # النموذج على المعالج قد يأخذ أكثر من دقيقة
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
EOF
ln -sf /etc/nginx/sites-available/chafibot /etc/nginx/sites-enabled/chafibot
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

if [ -n "$DOMAIN" ]; then
  apt-get install -y certbot python3-certbot-nginx
  if [ -n "$EMAIL" ]; then MAIL_ARG=(-m "$EMAIL"); else MAIL_ARG=(--register-unsafely-without-email); fi
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --redirect "${MAIL_ARG[@]}" || \
    echo "⚠️  فشلت شهادة HTTPS: تأكد أن $DOMAIN يشير إلى IP هذا الـ VPS ثم أعد: certbot --nginx -d $DOMAIN"
fi

echo
echo "================ انتهى التثبيت ================"
if grep -q '^DB_PASSWORD=$' "$APP_DIR/.env"; then
  echo "الخطوة التالية:"
  echo "  1) أكمل معلومات قاعدة البيانات:   nano $APP_DIR/.env"
  echo "  2) ابنِ الفهرس وشغّل الخدمة:       bash $APP_DIR/deploy/rebuild-index.sh"
else
  bash "$APP_DIR/deploy/rebuild-index.sh"
fi
echo
echo "في .env الخاص بـ Laravel (Hostinger):"
if [ -n "$DOMAIN" ]; then echo "  AI_SERVICE_URL=https://$DOMAIN"; else echo "  AI_SERVICE_URL=http://$(curl -s -4 ifconfig.me)"; fi
grep '^AI_SERVICE_KEY=' "$APP_DIR/.env" | sed 's/^/  /'
echo "ثم: php artisan config:clear"
