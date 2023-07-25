from PIL import Image

# 加载图像
image_path = "./img/13_result.jpg"
image = Image.open(image_path)

# 获取图像的分辨率（每英寸像素数）
resolution = image.info.get("dpi")
print("图像分辨率：", resolution)
