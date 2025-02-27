import matplotlib.pyplot as plt
from skimage import data,color,morphology
import cv2

# 输入图片路径
input_image_path = "/root/autodl-tmp/step5_largest_component.png"

# 读取图片
img = cv2.imread(input_image_path)
# #生成二值测试图像
# img=color.rgb2gray(data.horse())
# img=(img<0.5)*1

chull = morphology.convex_hull_image(img)

# 绘制轮廓并保存图片
fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax0, ax1 = axes.ravel()
ax0.imshow(img, plt.cm.gray)
ax0.set_title('original image')
ax1.imshow(chull,plt.cm.gray)
ax1.set_title('convex_hull image')

# 保存图片
plt.savefig('output_image.png')