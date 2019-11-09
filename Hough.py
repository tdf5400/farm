import cv2
import numpy as np
import math
import time
time_start = time.perf_counter()

"""
读取图像 与 初步处理
"""
img = cv2.resize(cv2.imread('./testLib/Low/3.jpg', cv2.IMREAD_GRAYSCALE), (640, 480))
cv2.imshow('Step1.Raw', img)

blur = img
blur = cv2.medianBlur(img, 3)               # 中值滤波
# blur = cv2.GaussianBlur(img, (3, 3), 0)   # 高斯滤波
cv2.imshow('Step2.Blur', blur)

"""
边界处理
"""
edge = cv2.Canny(blur, 150, 220)    # canny边缘检测
# edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)     # 自适应阈值

cv2.imshow('Step3.Edge', edge)

"""
Hough处理
"""
lines = cv2.HoughLines(edge, 1, np.pi / 180 * 2, 110)
if lines is not None:
    hough = np.zeros(edge.shape, dtype=np.uint8)
    line_direct_theta = 0   # 记录角度
    line_direct_times = 0   # 取平均用
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # y0=kx0+b    rho=
        pt1 = (int(x0 + 600*(-b)), int(y0 + 600 *(a)))
        pt2 = (int(x0 - 600 * (-b)), int(y0 - 600 * (a)))

        if not theta == 0:
            lineTheta = math.atan(-1 / math.tan(theta)) / np.pi * 180   # 直线角度
        else:
            lineTheta = 0
#         # 显示特定角度的直线
        if not -15 < lineTheta < 15 or 75 < line_direct_theta < 105:
            cv2.line(img, pt1, pt2, 255, 3, cv2.LINE_AA)
            line_direct_theta += theta
            line_direct_times += 1
            print(rho, lineTheta)
    cv2.imshow('Setp4.Hough', img)
    # 计算指引线
    if line_direct_times is not 0:
        line_direct_theta = line_direct_theta / line_direct_times
        print(f'direct:{line_direct_theta / np.pi * 180}')

time_elapsed = (time.perf_counter() - time_start)
print("Time used:", time_elapsed)
cv2.waitKey(0)
cv2.destroyAllWindows()


