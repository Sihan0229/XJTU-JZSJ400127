# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 定义输入视频和背景图片路径
video_path = '/root/autodl-tmp/XJTU-JZSJ400127/inference/input/test3.mp4'
background_path = '/root/autodl-tmp/XJTU-JZSJ400127/inference/output/average_frame.png'
output_video_path = '/root/autodl-tmp/XJTU-JZSJ400127/inference/output/no_bg.mp4'

# 读取背景图片
background = cv2.imread(background_path)

# 打开原始视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open the video.")
    exit()

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度

# 定义视频输出格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'是mp4格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # 逐帧读取视频
    if not ret:
        break  # 如果视频读取完毕，退出循环

    # 确保背景和当前帧尺寸一致
    background_resized = cv2.resize(background, (frame_width, frame_height))

    # 将当前帧与背景进行减法
    frame_no_bg = cv2.absdiff(frame, background_resized)

    # 保存处理后的帧
    out.write(frame_no_bg)

# 释放视频读取和写入对象
cap.release()
out.release()

print(f"视频已保存到: {output_video_path}")
