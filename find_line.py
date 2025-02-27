import cv2
import numpy as np

# 输入图片路径
input_image_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/output/average_frame.png"

# 读取图片
image = cv2.imread(input_image_path)
if image is None:
    print("图片加载失败！")
    exit()

# 保存原始图片

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊来减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用自适应阈值分割道路区域
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
cv2.imwrite("/root/autodl-tmp/XJTU-JZSJ400127/step_4_threshold.png", thresh)
