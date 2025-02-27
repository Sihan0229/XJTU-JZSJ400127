import cv2
import numpy as np
# 从多帧叠加背景图到多边形角点输出
def clean_and_fit_polygons(input_image, output_path, epsilon_ratio=0.04, min_area=500, morph_kernel_size=30):
    """
    对连通域进行内部黑色部分腐蚀清理，并拟合粗糙多边形边界，输出顶点坐标
    :param image_path: 输入二值图像路径
    :param output_path: 输出结果路径
    :param epsilon_ratio: 控制多边形拟合的粗糙度（比例值，越大越粗糙）
    :param min_area: 过滤小轮廓的面积阈值
    :param morph_kernel_size: 腐蚀膨胀的核大小，越大处理越粗糙
    """
    # 读取图像
    image = input_image

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值分割道路区域
    image = cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    height, width = image.shape

    # 平滑处理高斯模糊，卷积核大小由图像尺寸决定
    kernel_size = int(width / 12)
    # 如果 kernel_size 是偶数，增加 1 使其为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    # 使用 GaussianBlur 进行模糊处理
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # 二值化处理（Otsu方法）+形态学操作（闭运算去噪）
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 核大小根据图像调整
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 提取最大连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景
    largest_component = (labels == max_label).astype(np.uint8) * 255
    # cv2.imwrite('step4_largest_component.png', largest_component)


    image = largest_component
    # 创建形态学操作的核
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    # 使用膨胀操作填补小孔洞
    dilated = cv2.dilate(image, kernel, iterations=1)

    # 使用腐蚀操作恢复边界
    cleaned = cv2.erode(dilated, kernel, iterations=1)

    # 保存清理后的图像
    #cv2.imwrite("cleaned_image.png", cleaned)

    # 提取轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建输出图像
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 存储所有多边形顶点坐标
    all_polygons = []

    for contour in contours:
        # 过滤掉小轮廓
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # 多边形拟合
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)  # 计算拟合参数
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 将拟合的多边形顶点按 x 坐标从小到大排序
        sorted_vertices = sorted(approx[:, 0, :], key=lambda x: x[0])

        # 存储多边形顶点
        all_polygons.append(sorted_vertices)

        # 打印顶点坐标（从左至右）
        print("多边形顶点（从左至右）:")
        for vertex in sorted_vertices:
            print(f"({vertex[0]}, {vertex[1]})")

        # 绘制多边形
        cv2.polylines(result, [approx], isClosed=True, color=(0, 0, 255), thickness=10)

    # 保存结果
    cv2.imwrite(output_path, result)

    return all_polygons  # 返回所有多边形顶点





# # 输入图片路径
# input_image_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/output/average_frame.png"

# # 读取图片
# image = cv2.imread(input_image_path)

# # 调用函数
# output_image_path = "/root/autodl-tmp/longest_boundaries_output1.png"

# all_polygons = clean_and_fit_polygons(image, output_image_path)

# # 打印所有多边形顶点
# print(all_polygons)