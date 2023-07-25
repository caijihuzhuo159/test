import cv2
import numpy as np


def harris_corner_detection(image, distance_threshold):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算Harris角点响应
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # 通过非极大值抑制选择角点
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # 寻找角点的坐标
    corners = cv2.findNonZero(dst)

    # 对角点进行距离过滤并融合为一个角点
    filtered_corners = []
    for corner in corners:
        x, y = corner.ravel()
        merged = False
        for cx, cy in filtered_corners:
            if np.linalg.norm(np.array([x, y]) - np.array([cx, cy])) < distance_threshold:
                merged = True
                cx = (cx + x) // 2
                cy = (cy + y) // 2
                break
        if not merged:
            filtered_corners.append((x, y))

    # 根据y坐标的大小将角点分为3类，并用不同颜色表示
    grouped_corners = [[], [], []]
    for corner in filtered_corners:
        x, y = corner
        if y < 150:
            grouped_corners[0].append((x, y))
        elif y < 500:
            grouped_corners[1].append((x, y))
        else:
            grouped_corners[2].append((x, y))

    # 绘制角点
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, corners in enumerate(grouped_corners):
        for corner in corners:
            x, y = corner
            if y < 500:  # 只绘制y坐标小于500的角点
                cv2.circle(image, (x, y), 3, colors[i], -1)

    return image


# 读取图像
image = cv2.imread('./img/13.jpg')

# 设置角点之间的最小距离阈值
distance_threshold = 24

# 进行Harris角点检测并将距离小于阈值的多个角点融合为一个角点
result = harris_corner_detection(image, distance_threshold)

# 显示结果
cv2.imshow('Harris Corner Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
