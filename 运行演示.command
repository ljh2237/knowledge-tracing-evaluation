#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  bash scripts/setup.sh
fi
.venv/bin/python main.py --overwrite
printf '%s\n' '结果在 outputs/demo/predictions.csv。按回车关闭。'
read -r _
