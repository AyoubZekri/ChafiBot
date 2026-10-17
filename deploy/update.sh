#!/usr/bin/env bash
# تحديث ChafiBot بعد رفع تعديلات إلى GitHub:
#   bash deploy/update.sh
# (لا يمس .env ولا الفهرس؛ إذا تغيّر build_index.py شغّل بعده deploy/rebuild-index.sh)
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER=chafibot
cd "$APP_DIR"

git pull --ff-only
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
sudo -H -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q -r requirements.txt
systemctl restart chafibot

for _ in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null; then
    curl -s http://127.0.0.1:8001/health; echo
    echo "✅ تم التحديث"
    exit 0
  fi
  sleep 2
done
echo "❌ الخدمة لم تعمل. السبب:  journalctl -u chafibot -n 50 --no-pager"
exit 1
