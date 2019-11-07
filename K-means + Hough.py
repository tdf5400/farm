import cv2
import numpy as np
import math
import time
time_start = time.perf_counter()

"""
读取图像 与 初步处理
"""
img = cv2.resize(cv2.imread('./testRoad_2.jpg'), (640, 480))
cv2.imshow('Step1.Raw', img)

img = cv2.medianBlur(img, 9)               # 中值滤波
# img = cv2.GaussianBlur(img, (17, 17), 0)   # 高斯滤波
cv2.imshow('Step1.Raw', img)
"""
K-Means处理
"""
# K-Means 处理彩色图像
Z = img.reshape((-1, 3))    # 将三维数组变成二维数组
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    # 标准
K = 3   # 聚类数量
ret, label, center = cv2.kmeans(Z, K, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
A = np.resize(label.ravel(), (img.shape[0], img.shape[1]))  # 录入k-means结果

# 获取地面区域组号
value = 0.0
temp = A[(img.shape[0]-20):(img.shape[0]-10), (img.shape[1]-10):(img.shape[1]+10)]
for i in np.resize(temp, 1):
    value += i/100
value = int(value*100)
# 给目标区域上色
newArea = np.zeros(img.shape[0:2], dtype=np.uint8)  # 存储包含路面的新区域
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if A.item((i, j)) == value:
            newArea.itemset((i, j), 255)
        # if A[i, j] == value:
        #     newArea[i, j] = 255
# 处理区域
kernel = np.ones((45, 45), dtype=np.uint8)
# kernel = np.array((), dtype=np.uint8)

newArea = cv2.morphologyEx(newArea, cv2.MORPH_OPEN, kernel)  # 开运算
newArea = cv2.morphologyEx(newArea, cv2.MORPH_CLOSE, kernel)  # 闭运算
# newArea = cv2.erode(newArea, kernel, cv2.MORPH_ERODE)   # 腐蚀
newArea = cv2.GaussianBlur(newArea, (27, 27), sigmaX=0, sigmaY=0)
cv2.imshow('Step2.newArea', newArea)
cv2.imshow('K-means', cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0.2, newArea, 0.8, -3))    # ///////
"""
Canny处理
"""
# newArea = cv2.morphologyEx(newArea, cv2.MORPH_OPEN, kernel)
# newArea = cv2.Canny(newArea, 127, 127)
canny = cv2.Canny(newArea, 150, 200)

cv2.imshow('Step3.Canny', canny)
"""
每行白点累加
"""
# # 计算每行白点
# canny_line = np.zeros(canny.shape, dtype=np.uint8)
# for i in range(canny.shape[0]):
#     times = 0
#     value = 0
#     for j in range(canny.shape[1]):
#         if canny.item((i, j)) == 255:
#             value += j
#             times += 1
#     if times is not 0:
#         value = int(value / times)
#         canny_line.itemset((i, value), 255)
# cv2.imshow('Canny_Line', canny_line)

"""
Hough处理
"""
lines = cv2.HoughLines(canny, 1, np.pi / 180 * 4, 45)
if lines is not None:
    hough = np.zeros(canny.shape, dtype=np.uint8)
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
#         if not -15 < lineTheta < 15:
        if True:    #///////////
            cv2.line(img, pt1, pt2, 255, 3, cv2.LINE_AA)
            print(rho, lineTheta)
    cv2.imshow('Setp4.Hough', img)

time_elapsed = (time.perf_counter() - time_start)
print("Time used:", time_elapsed)
cv2.waitKey(0)
cv2.destroyAllWindows()
