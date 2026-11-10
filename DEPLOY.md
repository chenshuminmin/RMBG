# 白底图服务（BiRefNet-HRSOD）

FastAPI 单图白底图服务。宿主机端口 **6574** → 容器内 **8000**。结果全部在内存中返回，不落盘。

## 项目结构（已清理为最小集）

```
RMBG/
├── server.py               # 服务全部代码（153 行）
├── requirements.txt        # 运行依赖（不含 torch）
├── Dockerfile              # 公开 CUDA 基座
├── docker-compose.yaml     # 6574:8000 + resources 只读挂载
├── start_service.bat       # Windows 原生启动（6574）
├── allow_lan.bat           # 放行局域网入站（需 UAC）
├── _dl/fetch_model.py      # 模型分段下载工具（可选）
├── resources/              # 模型权重（不入库）
└── .venv/                  # 本机虚拟环境（不入库）
```

## 1. 模型

```bash
python -m pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com        # Windows PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
hf download ZhengPeng7/BiRefNet-HRSOD --local-dir resources/BiRefNet-HRSOD
```

`resources/BiRefNet-HRSOD/` 需含 `config.json` + `model.safetensors`（约 885 MB），另有模型自带的 `birefnet.py` / `BiRefNet_config.py`（服务通过 `trust_remote_code` 直接加载，不需要另装 BiRefNet 仓库源码）。

> hf-mirror 单线程约 0.5 MB/s。可用 `_dl/fetch_model.py` 16 线程分段下载：
> `python _dl/fetch_model.py ZhengPeng7/BiRefNet-HRSOD model.safetensors resources/BiRefNet-HRSOD/model.safetensors 16`

## 2. Docker

```bash
docker compose up -d --build
docker compose logs -f rmbg
curl http://127.0.0.1:6574/health
```

基座 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 自带 CUDA 版 torch/torchvision，所以 `requirements.txt` 里**故意不含 torch**——用 pip 装会覆盖成 CPU 版。`resources/` 只读挂载到 `/app/resources`，镜像不含权重。

需要 GPU：宿主机装 NVIDIA Container Toolkit。无 GPU 时删掉 compose 的 `deploy.resources` 段，服务退回 CPU（单图约 5~10 秒）。

## 3. Windows 原生（本机无 Docker）

```bash
python -m venv .venv
# 先装 CUDA 版 torch（官方源慢，可用国内镜像的直链 wheel）
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
start_service.bat        # 监听 6574
```

停止：`netstat -ano | findstr :6574` 找到 PID 后 `taskkill /PID <pid> /F`。

## 4. 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/process-single-image` | form-data 字段名 `file`（JPG/PNG），返回白底图 PNG |
| GET | `/health` | 健康检查，返回 status / device / model_dir |
| GET | `/docs` | FastAPI 在线接口文档，可直接上传图片试用 |

```bash
curl -X POST http://127.0.0.1:6574/process-single-image -F "file=@1.jpg" -o white_bg.png
```

## 5. 同一 WiFi 下让笔记本访问

服务监听 `0.0.0.0:6574`，本机 WiFi 网卡 IP 为 `192.168.1.14`。唯一障碍是 Windows 防火墙：该 WiFi 被识别为「公用网络」，默认拦截所有入站连接。

**双击 `allow_lan.bat` → UAC 点「是」**，创建一条规则：仅 TCP 6574 入站、仅允许本地子网 `LocalSubnet`、覆盖 Public/Private/Domain。不需要把网络改成「专用网络」。管理员 PowerShell 里也可直接执行：

```powershell
New-NetFirewallRule -DisplayName "RMBG WhiteBg TCP 6574" -Direction Inbound -Protocol TCP -LocalPort 6574 -Action Allow -Profile Any -RemoteAddress LocalSubnet
```

笔记本侧：

```
http://192.168.1.14:6574/health
http://192.168.1.14:6574/docs
POST http://192.168.1.14:6574/process-single-image   # form-data: file
```

```bash
curl http://192.168.1.14:6574/health
curl -X POST http://192.168.1.14:6574/process-single-image -F "file=@1.jpg" -o white_bg.png
```

排查顺序：① 笔记本 `ping 192.168.1.14` 不通 → 路由器开了「AP 隔离 / 访客网络」，或本机 IP 已变（`ipconfig` 重查）；② 本机 `netstat -ano | findstr :6574` 无 LISTEN → 服务停了，跑 `start_service.bat`；③ 第三方杀软防火墙单独放行 6574；④ 确认两边连的是同一个 WiFi。

## 6. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `./resources/BiRefNet-HRSOD` | 模型目录，容器内为 `/app/resources/BiRefNet-HRSOD` |
| `MODEL_REPO` | `ZhengPeng7/BiRefNet-HRSOD` | 本地模型缺失时的在线兜底 |
| `PORT` | `8000` | 容器内监听端口；本机启动由 `start_service.bat` 设为 6574 |
| `DEVICE` | `auto` | 可设 `cpu` / `cuda` 强制指定 |
