#!/usr/bin/env bash
# 独立虚拟环境复用本机已有的 PyTorch / CUDA，不修改系统安装的包。
set -e
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$app_dir"
python3 -m venv --without-pip --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
echo "环境已准备好。运行 bash start.sh 启动。"
