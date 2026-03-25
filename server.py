from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
import time
import uuid
from pathlib import Path


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
        print(f"✅ BiRefNet初始化完成\n📌 运行设备：{self.device}\n📂 临时结果目录：{self.temp_dir}")

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

        # 5. 生成临时文件名（UUID确保唯一性，避免覆盖）
        temp_filename = f"white_bg_{uuid.uuid4()}.png"
        temp_save_path = extractor.temp_dir / temp_filename

        # 6. 保存结果（PNG格式，避免JPG压缩失真）
        result_img.save(temp_save_path, format="PNG")
        print(f"📊 图像处理完成\n📄 原始文件：{file.filename}\n⏱️  耗时：{cost_time:.2f}s\n💾 临时保存路径：{temp_save_path}")

        # 7. 返回白底图（指定下载文件名，方便前端处理）
        return FileResponse(
            path=temp_save_path,
            filename=f"white_bg_{os.path.splitext(file.filename)[0]}.png",  # 下载文件名：white_bg_原始名.png
            media_type="image/png"
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
        return {
            "status": "healthy",
            "message": "服务正常运行，可接收单图处理请求",
            "device": extractor.device,
            "temp_dir": str(extractor.temp_dir)
        }
    return {
        "status": "unhealthy",
        "message": "服务未初始化成功，请检查模型或重启服务",
        "device": "unknown"
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