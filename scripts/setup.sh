#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then
  .venv/bin/python -c 'import sys; assert sys.version_info >= (3, 12), "已有环境需要 Python 3.12 或更新版本"'
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -r requirements-lock.txt
  else
    .venv/bin/python -m pip install -r requirements-lock.txt
  fi
elif command -v uv >/dev/null 2>&1 && [ -z "${PYTHON_BIN:-}" ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements-lock.txt
else
  task_python="${PYTHON_BIN:-python3.12}"
  "$task_python" -c 'import sys; assert sys.version_info >= (3, 12), "请使用 Python 3.12 或更新版本"'
  "$task_python" -m venv .venv
  .venv/bin/python -m pip install -r requirements-lock.txt
fi
printf '%s\n' '环境已安装。运行：source .venv/bin/activate && python main.py'
