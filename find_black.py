import cv2
import numpy as np

# Step 1: 读取灰度图像
image = cv2.imread('/root/autodl-tmp/XJTU-JZSJ400127/inference/output/line/step_4_threshold.png', cv2.IMREAD_GRAYSCALE)
height, width = image.shape
print(f"图像尺寸: {width}x{height}")

# Step 2: 平滑处理（高斯模糊）
# 计算卷积核大小
kernel_size = int(width / 20)
# 如果 kernel_size 是偶数，增加 1 使其为奇数
if kernel_size % 2 == 0:
    kernel_size += 1
# 使用 GaussianBlur 进行模糊处理
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Step 3: 二值化处理（Otsu方法）
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: 形态学操作（闭运算去噪）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 核大小根据图像调整
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Step 5: 连通域分析，提取最大连通区域
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景
largest_component = (labels == max_label).astype(np.uint8) * 255
cv2.imwrite('step5_largest_component.png', largest_component)

# Step 6: 提取最大区域的边界
contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # 用绿色画出边界
cv2.imwrite('step6_final_output.png', output)

