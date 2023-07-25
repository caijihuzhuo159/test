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
            if y < 150 and np.linalg.norm(np.array([x, y]) - np.array([cx, cy])) < distance_threshold:
                merged = True
                cx = (cx + x) // 2
                cy = (cy + y) // 2
                break
            elif y >= 150 and abs(x - cx) < 15 and abs(y - cy) < 55:
                merged = True
                if y < cy:
                    cx = x
                    cy = y
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

    # 重新排列角点按照x轴坐标的大小
    for corners in grouped_corners:
        corners.sort(key=lambda corner: corner[0])

    print(len(grouped_corners[0]))
    print(len(grouped_corners[1]))
    print(grouped_corners[0])
    print(grouped_corners[1])

    if len(grouped_corners[0]) != len(grouped_corners[1]):
        print("阴阳极极点不成对")
        return image
    else:
        for i in range(len(grouped_corners[0])):
            point1 = grouped_corners[0][i]  # grouped_corners[0]的第i个点
            point2 = grouped_corners[1][i]  # grouped_corners[1]的第j个点

            vertical_distance = abs(point1[1] - point2[1])  # 计算垂直距离
            print("垂直距离:", vertical_distance)

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
