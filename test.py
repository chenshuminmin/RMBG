import requests
from pathlib import Path

def upload_image_to_birefnet(
    image_path: str = "/app/BiRefNet/imgs/O1CN01Ax98l21Zm8aJisVX4_!!1752583236-0-cib.jpg",
    save_path: str = "./white_bg_output.png",
    api_url: str = "http://10.8.28.109:6574/process-single-image"
) -> bool:
    """
    上传图像到 BiRefNet 白底图服务，保存结果到指定路径
    :param image_path: 待上传的本地图像路径（与 curl 中的 @ 后路径一致）
    :param save_path: 白底图保存路径（与 curl 中的 -o 路径一致）
    :param api_url: 服务接口地址（固定）
    :return: 成功返回 True，失败返回 False
    """
    # 1. 验证本地图像是否存在（避免 curl: (26) 类似错误）
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"❌ 错误：本地图像不存在 → {image_path}")
        return False
    if not img_file.is_file():
        print(f"❌ 错误：{image_path} 不是有效文件（可能是目录）")
        return False

    # 2. 构造请求（模拟 curl -F "file=@xxx" 的表单上传）
    # 关键：files 参数格式为 {"file": (文件名, 文件流, MIME类型)}，与服务端接口参数名一致
    try:
        with open(img_file, "rb") as f:
            files = {
                "file": (
                    img_file.name,  # 上传时的文件名（服务端可识别）
                    f,              # 图像二进制流
                    "image/jpeg"    # MIME类型（JPG格式固定为 image/jpeg，PNG为 image/png）
                )
            }

            print(f"📤 正在上传图像：{image_path}")
            print(f"🎯 目标服务接口：{api_url}")
            
            # 发送 POST 请求（模拟 curl -X POST，超时时间30秒避免卡住）
            response = requests.post(
                url=api_url,
                files=files,
                timeout=30  # 与 curl 默认超时逻辑一致，避免网络异常导致无限等待
            )

    except Exception as e:
        # 捕获本地文件读写异常（如权限不足）或网络连接异常（如服务未启动）
        print(f"❌ 请求发送失败：{str(e)}")
        return False

    # 3. 处理响应结果
    # 成功：服务端返回 200 OK，响应体为 PNG 图像二进制流
    if response.status_code == 200:
        try:
            # 保存响应到本地（模拟 curl -o ./white_bg_output.png）
            with open(save_path, "wb") as f:
                f.write(response.content)  # 直接写入二进制流（保留图像完整性）
            
            # 验证保存结果
            save_file = Path(save_path)
            if save_file.exists() and save_file.stat().st_size > 0:
                print(f"✅ 成功！白底图已保存到 → {save_path}")
                print(f"📊 图像大小：{save_file.stat().st_size / 1024:.2f} KB")
                return True
            else:
                print(f"❌ 保存失败：{save_path} 生成后为空或不存在")
                return False

        except Exception as e:
            print(f"❌ 保存白底图时出错：{str(e)}（可能是保存路径无权限）")
            return False

    # 失败：服务端返回错误状态码（如 400 格式错误、500 服务异常）
    else:
        print(f"❌ 服务端返回错误：状态码 {response.status_code}")
        # 尝试解析服务端返回的 JSON 错误信息（与 FastAPI 接口错误格式匹配）
        try:
            error_detail = response.json().get("detail", "未返回具体错误信息")
            print(f"📝 错误详情：{error_detail}")
        except:
            # 若响应不是 JSON（如服务崩溃返回 HTML），直接打印原始响应
            print(f"📝 原始错误响应：{response.text[:200]}...")  # 只显示前200字符避免过长
        return False


# --------------------------
# 执行入口（直接运行脚本即可）
# --------------------------
if __name__ == "__main__":
    # 调用函数，参数与 curl 命令完全对应
    success = upload_image_to_birefnet(
        image_path="/data/tyx/test/RMBG/tests/1.jpg",
        save_path="./white_bg_output.png"
    )
    # 根据执行结果退出（0=成功，1=失败，符合 Unix 命令行习惯）
    exit(0 if success else 1)
