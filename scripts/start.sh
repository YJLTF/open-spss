#!/usr/bin/env bash
# OpenSPSS 一体化启动脚本:自动创建虚拟环境、安装依赖并启动服务
# 用法: bash scripts/start.sh [端口]
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-8420}"

if [ ! -d .venv ]; then
  echo ">> 创建虚拟环境 .venv ..."
  python3 -m venv .venv
fi

PY=".venv/bin/python"
if ! "$PY" -c "import fastapi, pandas, statsmodels, sklearn" 2>/dev/null; then
  echo ">> 安装依赖 ..."
  "$PY" -m pip install --upgrade pip -q
  "$PY" -m pip install -e . -q
fi

echo ">> 启动 OpenSPSS: http://127.0.0.1:${PORT}"
exec "$PY" -m open_spss --port "$PORT"
