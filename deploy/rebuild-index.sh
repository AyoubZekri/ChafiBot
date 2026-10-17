#!/usr/bin/env bash
# يبني فهرس البحث من قاعدة بيانات Chafi ثم يعيد تشغيل الخدمة.
# يعمل وحده كل ليلة (chafibot-index.timer)، ويمكن تشغيله يدوياً بعد إضافة محتوى:
#   bash deploy/rebuild-index.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER=chafibot
cd "$APP_DIR"

echo "==> بناء الفهرس (قد يأخذ بضع دقائق)"
# البناء في مجلد مؤقت ثم الاستبدال: إذا فشل يبقى الفهرس القديم يعمل
sudo -H -u "$APP_USER" env INDEX_DIR_OVERRIDE="$APP_DIR/index.new" \
  "$APP_DIR/.venv/bin/python" build_index.py
rm -rf "$APP_DIR/index.old"
[ -d "$APP_DIR/index" ] && mv "$APP_DIR/index" "$APP_DIR/index.old"
mv "$APP_DIR/index.new" "$APP_DIR/index"
rm -rf "$APP_DIR/index.old"

echo "==> إعادة تشغيل الخدمة"
systemctl restart chafibot
for _ in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null; then
    curl -s http://127.0.0.1:8001/health; echo
    echo "✅ ChafiBot يعمل"
    exit 0
  fi
  sleep 2
done
echo "❌ الخدمة لم تعمل. السبب:  journalctl -u chafibot -n 50 --no-pager"
exit 1
