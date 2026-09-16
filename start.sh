#!/usr/bin/env bash
# 启动本地服务；重复启动会打开已经运行的同一实例。
set -e
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$app_dir"
if [ ! -x .venv/bin/python ]; then
    echo "请先在项目目录运行：bash setup.sh"
    exit 1
fi
mkdir -p data
exec .venv/bin/python main.py --open "$@" >> data/server.log 2>&1
