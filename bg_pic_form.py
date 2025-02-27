import cv2
import numpy as np

# 输入视频路径
input_video_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/input/test3.mp4"
# 输出图片路径
output_image_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/output/average_frame.png"

# 打开输入视频
cap = cv2.VideoCapture(input_video_path)

# 获取视频的基本信息
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 视频高度

# 初始化一个数组用于累加所有帧
sum_frames = np.zeros((height, width, 3), dtype=np.float32)

# 遍历视频帧
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 累加帧
    sum_frames += frame.astype(np.float32)
    count += 1

# 释放视频资源
cap.release()

# 计算平均帧
average_frame = sum_frames / count

# 将平均帧转换为uint8格式（像素值范围0-255）
average_frame = np.clip(average_frame, 0, 255).astype(np.uint8)

# 保存平均帧为图片
cv2.imwrite(output_image_path, average_frame)

print(f"平均图片已保存：{output_image_path}")
