#!/bin/bash
set -euo pipefail

# roslogging.py を 0644 (rw-r--r--) で配置する
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/roslogging.py"
DST_DIR="/opt/ros/noetic/lib/python3/dist-packages/rosgraph"
DST_FILE="$DST_DIR/roslogging.py"

# ディレクトリを作成し、権限 644 でコピー
if command -v install >/dev/null 2>&1; then
	# install はモードを明示できるため推奨
	install -m 644 -D "$SRC" "$DST_FILE"
else
	mkdir -p "$DST_DIR"
	cp "$SRC" "$DST_FILE"
	chmod 644 "$DST_FILE"
fi

echo "Installed $SRC to $DST_FILE with mode 0644"