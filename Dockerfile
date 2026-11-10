# 白底图提取服务镜像
# 基座改为公开镜像（原 registry.linker.cc/linker/bg-base 需要私有 registry 权限），
# 已自带 torch 2.5.1 + CUDA 12.4，避免在构建阶段下载 2.5GB 的 torch 轮子。
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    PORT=8000 \
    MODEL_PATH=/app/resources/BiRefNet-HRSOD

# 先装非 torch 依赖（torch/torchvision 由基座提供 CUDA 版，不能用 pip 覆盖）
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

# 只拷服务代码，模型权重由 volume 挂载到 /app/resources（不打进镜像）
COPY server.py /app/server.py

EXPOSE 8000
CMD ["python", "server.py"]
