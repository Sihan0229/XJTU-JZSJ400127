import cv2
import numpy as np

# 输入和输出视频路径
input_video_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/input/test3.mp4"
output_video_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/output/50_averaged_video.mp4"

# 打开输入视频
cap = cv2.VideoCapture(input_video_path)

# 获取视频的基本信息
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 初始化一个缓冲区用于存储帧
frame_buffer = []

# 遍历视频帧
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将当前帧加入缓冲区
    frame_buffer.append(frame.astype(np.float32))  # 转为float32以便计算平均

    # 如果帧数小于等于10，取当前所有帧的平均
    if i < 50:
        avg_frame = np.mean(frame_buffer, axis=0)
    else:
        # 从第10帧开始，取当前帧的前10帧的平均
        avg_frame = np.mean(frame_buffer[-50:], axis=0)

    # 将平均帧转换回uint8格式（像素值范围0-255）
    avg_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)

    # 写入到输出视频
    out.write(avg_frame)

    print('Done. frame_count=' , i )

# 释放资源
cap.release()
out.release()

print(f"新视频已生成：{output_video_path}")
