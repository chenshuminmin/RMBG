from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
import torch
from torchvision import transforms
import os
import shutil
import numpy as np
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional


# 结果是否落盘。默认关闭：容器内每请求生成一个 PNG 且从不清理，会把容器
# 可写层写满（表现为 "[Errno 28] No space left on device"，所有请求 500）。
# 现在默认在内存中直接返回，磁盘不再是处理链路的一部分。
SAVE_TEMP_RESULTS = os.getenv("SAVE_TEMP_RESULTS", "0").strip().lower() in {"1", "true", "yes"}
# 落盘模式下：可用空间低于该值时先清理再放弃落盘（不失败）。
MIN_FREE_BYTES_BEFORE_SAVE = 512 * 1024 * 1024
# 启动时自动清理超过该小时数的历史结果。
STARTUP_PURGE_MAX_AGE_HOURS = float(os.getenv("TEMP_RESULT_MAX_AGE_HOURS", "24"))


def _disk_free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return -1


def _purge_old_results(temp_dir: Path, max_age_hours: float = 1.0, keep_recent: int = 0) -> tuple[int, int]:
    """删除过期的历史结果文件，返回 (删除数量, 释放字节数)。"""
    removed = 0
    freed = 0
    try:
        entries = [p for p in temp_dir.glob("white_bg_*") if p.is_file()]
    except OSError:
        return 0, 0
    if keep_recent:
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        entries = entries[keep_recent:]
    cutoff = time.time() - max(0.0, max_age_hours) * 3600
    for path in entries:
        try:
            if max_age_hours > 0 and path.stat().st_mtime > cutoff:
                continue
            size = path.stat().st_size
            path.unlink()
            removed += 1
            freed += size
        except OSError:
            continue
    return removed, freed


def _safe_save_result(payload: bytes, temp_dir: Path) -> Optional[Path]:
    """尽量把结果落到磁盘，任何空间问题都只告警、绝不让请求失败。"""
    if _disk_free_bytes(temp_dir) < MIN_FREE_BYTES_BEFORE_SAVE:
        removed, freed = _purge_old_results(temp_dir, max_age_hours=1.0)
        print(f"🧹 磁盘空间不足，已清理历史结果 {removed} 个（释放 {freed / 1024 / 1024:.1f} MB）")
        if _disk_free_bytes(temp_dir) < MIN_FREE_BYTES_BEFORE_SAVE:
            print("⚠️ 磁盘空间仍不足，本次结果不落盘（不影响返回）")
            return None
    try:
        path = temp_dir / f"white_bg_{uuid.uuid4()}.png"
        path.write_bytes(payload)
        return path
    except OSError as exc:
        print(f"⚠️ 结果落盘失败（不影响返回）：{exc}")
        return None


# 初始化FastAPI应用（仅单图处理相关配置）
app = FastAPI(
    title="BiRefNet白底图提取服务",
    description="仅支持单图上传处理，返回白底前景图（支持JPG/PNG格式）",
    version="1.0"
)


class BiRefNetWhiteBgExtractor:
    """精简版BiRefNet提取器（仅保留单图处理核心逻辑）"""
    def __init__(self, device="auto"):
        # 1. 设备初始化（优先GPU，无则自动切换CPU）
        self.device = self._setup_device(device)
        # 2. 加载模型与预处理流水线（全局仅初始化1次）
        self.model, self.transform = self._load_model()
        # 3. 创建临时结果目录（存储处理后的白底图，避免内存堆积）
        self.temp_dir = Path("./temp_white_bg_results")
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        removed, freed = _purge_old_results(self.temp_dir, max_age_hours=STARTUP_PURGE_MAX_AGE_HOURS)
        if removed:
            print(f"🧹 启动时清理历史结果 {removed} 个（释放 {freed / 1024 / 1024:.1f} MB）")
        print(f"✅ BiRefNet初始化完成\n📌 运行设备：{self.device}\n📂 临时结果目录：{self.temp_dir}"
              f"\n💾 结果落盘：{'开启' if SAVE_TEMP_RESULTS else '关闭（内存直接返回）'}")

    def _setup_device(self, device):
        """简化设备选择：仅支持auto/cpu/cuda"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device if device in ["cpu", "cuda"] else "cpu"

    def _load_model(self):
        """模型加载（含关键错误捕获，提示更明确）"""
        # 导入BiRefNet模型（确保models/birefnet.py在当前目录）
        import sys
        sys.path.insert(0, "./")
        try:
            from BiRefNet.models.birefnet import BiRefNet
        except ImportError:
            raise ImportError("❌ 未找到模型文件：请将models/birefnet.py放在当前运行目录")

        # 加载预训练权重（处理网络/权重地址问题）
        try:
            model = BiRefNet.from_pretrained("zhengpeng7/BiRefNet-HRSOD")
            model.to(self.device).eval()  # 直接切换推理模式，禁用梯度计算
        except Exception as e:
            raise RuntimeError(f"❌ 模型权重加载失败：{str(e)}\n请检查网络连接或确认Hugging Face权重地址有效")

        # 图像预处理（保持原效果，无冗余步骤）
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # 模型输入尺寸，确保推理效果
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return model, transform

    def _generate_white_bg(self, image: Image.Image, pred_mask: Image.Image) -> Image.Image:
        """核心：生成白底前景图（无冗余计算）"""
        # 掩码适配原始图像尺寸
        pred_mask_resized = pred_mask.resize(image.size)
        # 掩码归一化（0=背景，1=前景）
        mask_np = np.array(pred_mask_resized, dtype=np.float32) / 255.0
        # 原始图像转数组（RGB格式）
        image_np = np.array(image.convert("RGB"), dtype=np.float32)
        # 融合前景与白色背景（公式简化，避免中间变量）
        white_bg = np.ones_like(image_np) * 255.0  # 白色背景（RGB:255,255,255）
        refined_np = (image_np * mask_np[..., np.newaxis]) + (white_bg * (1 - mask_np[..., np.newaxis]))
        # 格式转换（直接裁剪+uint8，符合图像标准）
        return Image.fromarray(np.clip(refined_np, 0, 255).astype(np.uint8))

    def process_single_image(self, image: Image.Image) -> tuple[Image.Image, float]:
        """处理单张PIL图像，返回白底图和处理耗时"""
        start_time = time.time()
        # 1. 图像预处理（强制RGB，避免灰度图/透明通道干扰）
        image_rgb = image.convert("RGB")
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)  # 增加批次维度

        # 2. 模型推理（混合精度+无梯度，加速并降低显存占用）
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16), torch.no_grad():
            pred_np = self.model(input_tensor)[-1].sigmoid()[0].squeeze().cpu().numpy()

        # 3. 生成白底图
        pred_mask = transforms.ToPILImage()(torch.tensor(pred_np))  # 掩码转PIL
        white_bg_img = self._generate_white_bg(image_rgb, pred_mask)

        # 4. 计算耗时
        cost_time = time.time() - start_time
        return white_bg_img, cost_time


# 全局初始化提取器（单例模式，避免重复加载模型浪费资源）
try:
    extractor = BiRefNetWhiteBgExtractor(device="auto")
except Exception as e:
    print(f"❌ 服务初始化失败：{str(e)}")
    extractor = None  # 标记为未初始化，后续接口会返回500错误


# --------------------------
# 核心接口：单图上传处理
# --------------------------
@app.post(
    path="/process-single-image",
    response_class=FileResponse,
    summary="上传单张图像，返回白底图",
    description="支持JPG/JPEG/PNG格式，返回处理后的PNG格式白底图"
)
async def process_single_image(file: UploadFile = File(..., description="待处理的图像文件（JPG/PNG）")):
    # 1. 检查服务是否初始化成功
    if extractor is None:
        raise HTTPException(
            status_code=500,
            detail="服务未初始化成功，请检查模型文件或网络连接后重启服务"
        )

    # 2. 检查文件格式（仅允许JPG/PNG）
    allowed_content_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式：{file.content_type}\n仅允许上传JPG（image/jpeg）或PNG（image/png）格式"
        )

    try:
        # 3. 读取上传的图像（避免文件流关闭导致的错误）
        with Image.open(file.file) as img:
            # 4. 处理图像（调用提取器核心方法）
            result_img, cost_time = extractor.process_single_image(img)

        # 5. 结果编码到内存（PNG格式，避免JPG压缩失真）
        #    默认不落盘：容器可写层曾因历史结果堆积被写满，导致所有请求
        #    "[Errno 28] No space left on device"。内存返回让磁盘不再是瓶颈。
        with BytesIO() as buffer:
            result_img.save(buffer, format="PNG")
            payload = buffer.getvalue()

        if SAVE_TEMP_RESULTS:
            temp_save_path = _safe_save_result(payload, extractor.temp_dir)
            if temp_save_path is not None:
                print(f"💾 临时保存路径：{temp_save_path}")

        # 6. 返回白底图（指定下载文件名，方便前端处理）
        download_name = f"white_bg_{os.path.splitext(file.filename or 'image')[0]}.png"
        print(f"📊 图像处理完成\n📄 原始文件：{file.filename}\n⏱️  耗时：{cost_time:.2f}s\n📦 结果大小：{len(payload) / 1024:.1f} KB")

        return Response(
            content=payload,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )

    except Exception as e:
        # 捕获处理过程中的所有异常，返回明确错误信息
        error_msg = f"图像处理失败：{str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        # 确保文件流关闭，释放资源
        await file.close()


# --------------------------
# 辅助接口：服务健康检查
# --------------------------
@app.get(
    path="/health",
    summary="服务健康检查",
    description="检查服务是否正常运行，返回设备信息和状态"
)
async def health_check():
    if extractor is not None:
        free = _disk_free_bytes(extractor.temp_dir)
        return {
            "status": "healthy",
            "message": "服务正常运行，可接收单图处理请求",
            "device": extractor.device,
            "temp_dir": str(extractor.temp_dir),
            "save_temp_results": SAVE_TEMP_RESULTS,
            "free_bytes": free,
            "free_gb": round(free / 1024 ** 3, 2),
            "low_disk": 0 <= free < MIN_FREE_BYTES_BEFORE_SAVE,
        }
    return {
        "status": "unhealthy",
        "message": "服务未初始化成功，请检查模型或重启服务",
        "device": "unknown"
    }


# --------------------------
# 运维接口：清理历史临时结果
# --------------------------
@app.post(
    path="/cleanup",
    summary="清理历史白底结果文件",
    description="删除 temp_white_bg_results 下的历史结果，释放磁盘空间（磁盘写满时用于救急）"
)
async def cleanup_results(max_age_hours: float = 0.0):
    if extractor is None:
        raise HTTPException(status_code=500, detail="服务未初始化成功，无法清理")
    removed, freed = _purge_old_results(extractor.temp_dir, max_age_hours=max_age_hours)
    free = _disk_free_bytes(extractor.temp_dir)
    return {
        "removed": removed,
        "freed_bytes": freed,
        "freed_mb": round(freed / 1024 / 1024, 2),
        "free_bytes": free,
        "free_gb": round(free / 1024 ** 3, 2),
    }


# --------------------------
# 服务启动入口
# --------------------------
if __name__ == "__main__":
    import uvicorn
    # 启动服务（单worker避免多进程重复加载模型，降低显存占用）
    uvicorn.run(
        app="__main__:app",  # 指定FastAPI应用实例
        host="0.0.0.0",      # 监听所有网络接口，支持外部访问
        port=8000,           
        workers=1,           # 单进程，避免模型重复加载
        reload=False,        # 生产环境禁用自动重载（降低资源占用）
        log_level="info"     # 日志级别：info（显示关键信息，避免冗余）
    )
