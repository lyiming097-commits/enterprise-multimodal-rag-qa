# Python 运行时（前端在宿主机构建，产物直接复制，避免多阶段并发构建导致内存不足）
FROM python:3.12-slim AS app
WORKDIR /app

# 国内网络可用清华 PyPI 镜像加速；如需官方源，构建时传 --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# opencv / rapidocr 运行时系统依赖（改用清华 Debian 源，deb.debian.org 国内极慢且易卡死）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖清单与源码，利用 Docker 层缓存
# 注意：必须用 -e（editable）安装，app/api/main.py 依赖源码目录结构定位 frontend/dist
COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
# 先装 CPU 版 torch（走上海交大镜像），避免默认从 PyPI 拉取带 CUDA 的 torch 及其庞大 nvidia 依赖
RUN pip install --no-cache-dir torch \
    --index-url https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cpu \
    --extra-index-url "${PIP_INDEX_URL}"

# 再装其余依赖（清华源；torch 已就位，不会重复安装 CUDA 版）
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" -e ".[ocr]"

# 复制前端构建产物（在宿主机 npm run build 生成；对应 app/api/main.py 中 FRONTEND_DIST = ROOT/frontend/dist）
COPY frontend/dist ./frontend/dist

# 入口脚本
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
