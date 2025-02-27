import cv2
import numpy as np


def clean_and_fit_polygons(image_path, output_path, epsilon_ratio=0.04, min_area=500, morph_kernel_size=30):
    """
    对连通域进行内部黑色部分腐蚀清理，并拟合粗糙多边形边界，输出顶点坐标
    :param image_path: 输入二值图像路径
    :param output_path: 输出结果路径
    :param epsilon_ratio: 控制多边形拟合的粗糙度（比例值，越大越粗糙）
    :param min_area: 过滤小轮廓的面积阈值
    :param morph_kernel_size: 腐蚀膨胀的核大小，越大处理越粗糙
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 创建形态学操作的核
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    # 使用膨胀操作填补小孔洞
    dilated = cv2.dilate(image, kernel, iterations=1)

    # 使用腐蚀操作恢复边界
    cleaned = cv2.erode(dilated, kernel, iterations=1)

    # 保存清理后的图像
    cv2.imwrite("cleaned_image.png", cleaned)

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
    print(f"清理后的多边形拟合结果已保存到: {output_path}")

    return all_polygons  # 返回所有多边形顶点


# 调用函数
input_image_path = "/root/autodl-tmp/step5_largest_component.png"
output_image_path = "/root/autodl-tmp/longest_boundaries_output.png"
all_polygons = clean_and_fit_polygons(input_image_path, output_image_path)

# 打印所有多边形顶点
print("\n所有多边形顶点:")
for i, polygon in enumerate(all_polygons):
    print(f"多边形 {i+1}:")
    for vertex in polygon:
        print(f"({vertex[0]}, {vertex[1]})")