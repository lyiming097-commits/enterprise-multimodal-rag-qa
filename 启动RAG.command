#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
  echo "缺少 .env，请先从 .env.example 复制并填写 DEEPSEEK_API_KEY。"
  read -k 1 "?按任意键退出…"
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "首次启动：正在创建 Python 环境…"
  python3 -m venv .venv
  .venv/bin/pip install -e .
fi

if ! .venv/bin/python -c "import rapidocr_onnxruntime" >/dev/null 2>&1; then
  echo "正在安装扫描件 OCR 组件…"
  .venv/bin/pip install -e '.[ocr]'
fi

if [[ ! -f frontend/dist/index.html ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "缺少前端构建产物且未找到 npm，请先安装 Node.js。"
    read -k 1 "?按任意键退出…"
    exit 1
  fi
  echo "首次启动：正在构建 Vue 页面…"
  (cd frontend && npm install && npm run build)
fi

echo "正在初始化数据库…"
.venv/bin/python -m scripts.init_db

(sleep 2; open "http://127.0.0.1:8000") &
echo "RAG Studio 启动中：http://127.0.0.1:8000"
echo "关闭本窗口或按 Control+C 即可停止。"
.venv/bin/python -m scripts.run_app
