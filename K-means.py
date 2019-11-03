import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
start = time.perf_counter()

img = cv2.resize(cv2.imread('./testRoad_3.jpg'), (640, 480))
cv2.imshow('Raw', img)
# 对原图初步处理
# img = cv2.medianBlur(img, 23)       # 中值滤波
img = cv2.GaussianBlur(img, (17, 17), 0)   # 高斯滤波

# K-Means 处理彩色图像
Z = img.reshape((-1, 3))
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    # 标准
K = 3   # 聚类数量
ret, label, center = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
A = np.resize(label.ravel(), (img.shape[0], img.shape[1]))  # 录入k-means结果

# 获取地面区域组号
value = 0.0
temp = A[(img.shape[0]-20):(img.shape[0]-10), (img.shape[1]-10):(img.shape[1]+10)]
for i in np.resize(temp, 1):
    value += i/100
value = int(value*100)
# 给区域上色
newArea = np.zeros(img.shape, dtype=np.uint8)  # 存储包含路面的新区域
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if A[i, j] == value:
            newArea[i, j] = (255, 255, 255)
            img[i, j] = (255, 255, 255)
        else:
            pass
            # img[i, j] = (127, 127, 127)
        # elif A[i, j] == 0:
        #     img[i, j] = (127, 0, 0)
        # elif A[i, j] == 1:
        #     img[i, j] = (0, 127, 0)
        # elif A[i, j] == 2:
        #     img[i, j] = (0, 0, 127)
# 处理区域
kernel = np.ones((83, 83), dtype=np.uint8)
# newArea = cv2.erode(newArea, kernel, cv2.MORPH_ERODE)   # 腐蚀
# newArea = cv2.morphologyEx(newArea, cv2.MORPH_OPEN, kernel)  # 开运算
# img = cv2.Canny(img, 127, 255)
# 画集聚点
for i in range(K):
    img[(center[i, 0]-5): (center[i, 0]+5), (center[i, 1]-5): (center[i, 1]+5)] = (0, 0, 255)
    # img[center[:,0], center[:, 1]] = (0, 0, 255)
# newArea = cv2.morphologyEx(newArea, cv2.MORPH_OPEN, kernel)
# newArea = cv2.Canny(newArea, 127, 127)


cv2.imshow('Result', img)
cv2.imshow('NewArea', newArea)


elapsed = (time.perf_counter() - start)
print("Time used:", elapsed)
cv2.waitKey(0)
cv2.destroyAllWindows()
