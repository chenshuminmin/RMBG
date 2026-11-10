"""BiRefNet-HRSOD 白底图提取服务（单图上传，内存返回 PNG）。"""

import contextlib
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms

# 权重预先下载到 ./resources/BiRefNet-HRSOD（见 DEPLOY.md），
# 容器里通过 volume 挂载，运行期不联网。
MODEL_DIR = Path(os.getenv("MODEL_PATH", "./resources/BiRefNet-HRSOD")).expanduser()
MODEL_REPO = os.getenv("MODEL_REPO", "ZhengPeng7/BiRefNet-HRSOD")
DEVICE = os.getenv("DEVICE", "auto")


def _has_local_weights(model_dir: Path) -> bool:
    """配置 + 权重齐全才算本地可用，避免把半下载目录当成模型。"""
    if not model_dir.is_dir():
        return False
    has_conf = (model_dir / "config.json").is_file()
    has_weight = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    return has_conf and has_weight


# 本地权重齐全时强制离线，防止启动时回连 huggingface.co 卡住。
if _has_local_weights(MODEL_DIR) and "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

app = FastAPI(
    title="BiRefNet 白底图提取服务",
    description="上传单张 JPG/PNG，返回白底前景图 PNG",
    version="1.0",
)


class BiRefNetWhiteBgExtractor:
    def __init__(self, device: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
        if self.device not in {"cpu", "cuda"}:
            self.device = "cpu"
        self.model, self.transform = self._load_model()
        print(f"✅ 初始化完成 | 设备={self.device} | 模型={MODEL_DIR}", flush=True)

    def _load_model(self):
        """本地优先；用模型仓库自带的单文件 birefnet.py（trust_remote_code）。"""
        from transformers import AutoModelForImageSegmentation

        use_local = _has_local_weights(MODEL_DIR)
        source = str(MODEL_DIR) if use_local else MODEL_REPO
        if not use_local:
            print(f"⚠️ 本地模型缺失（{MODEL_DIR}），回退在线仓库 {MODEL_REPO}", flush=True)
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                source, trust_remote_code=True, local_files_only=use_local
            )
            return (
                model.to(self.device).eval(),
                transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"模型加载失败：{exc}\n请确认 {MODEL_DIR} 下有 config.json + model.safetensors，"
                f"或用 MODEL_PATH 指向正确目录"
            ) from exc

    def process(self, image: Image.Image) -> tuple[Image.Image, float]:
        """返回白底图与耗时（秒）。"""
        started = time.time()
        image_rgb = image.convert("RGB")
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # CPU 不支持 fp16 autocast，仅在 CUDA 上启用混合精度
        autocast = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self.device == "cuda" else contextlib.nullcontext()
        )
        with autocast, torch.no_grad():
            pred = self.model(input_tensor)[-1].sigmoid()[0].squeeze().float().cpu().numpy()

        mask = transforms.ToPILImage()(torch.from_numpy(pred)).resize(image_rgb.size)
        alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        foreground = np.asarray(image_rgb, dtype=np.float32)
        result = Image.fromarray(
            np.clip(foreground * alpha + 255.0 * (1 - alpha), 0, 255).astype(np.uint8)
        )
        return result, time.time() - started


# 全局单例，避免重复加载模型占用显存
try:
    extractor = BiRefNetWhiteBgExtractor(device=DEVICE)
except Exception as exc:
    print(f"❌ 服务初始化失败：{exc}", flush=True)
    extractor = None


@app.post("/process-single-image", summary="上传单张图像，返回白底图 PNG")
async def process_single_image(file: UploadFile = File(..., description="JPG/PNG 图像")):
    if extractor is None:
        raise HTTPException(500, "服务未初始化成功，请检查模型目录后重启")
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(400, f"不支持的格式：{file.content_type}，仅允许 JPG/PNG")

    try:
        with Image.open(file.file) as img:
            result_img, cost = extractor.process(img)

        with BytesIO() as buffer:
            result_img.save(buffer, format="PNG")
            payload = buffer.getvalue()

        stem = Path(file.filename or "image").stem
        print(f"📊 {file.filename} | {cost:.2f}s | {len(payload) / 1024:.1f} KB", flush=True)
        return Response(
            content=payload,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="white_bg_{stem}.png"'},
        )
    except Exception as exc:
        raise HTTPException(500, f"图像处理失败：{exc}") from exc
    finally:
        await file.close()


@app.get("/health", summary="健康检查")
async def health_check():
    if extractor is None:
        return {"status": "unhealthy", "message": "模型未加载成功", "device": "unknown"}
    return {"status": "healthy", "device": extractor.device, "model_dir": str(MODEL_DIR)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),  # 容器内 8000 → 宿主机 6574
        workers=1,
        reload=False,
        log_level="info",
    )
