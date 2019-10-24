import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.resize(cv2.imread('./testRoad.jpg'), (640, 480))
img = cv2.medianBlur(img, 13)   # 中值滤波
cv2.imshow('Demo', img)
# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center=cv2.kmeans(Z,K,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
A = np.resize(label.ravel(), (img.shape[0], img.shape[1]))

# 获取地面组组号
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
        elif A[i, j] == 0:
            img[i, j] = (127, 127, 127)
        elif A[i, j] == 1:
            img[i, j] = (127, 127, 127)
        elif A[i, j] == 2:
            img[i, j] = (127, 127, 127)
        # elif A[i, j] == 3:
        #     img[i, j] = (127, 127, 127)
# 进行腐蚀
kernel = np.ones((23, 23), dtype=np.uint8)
newArea = cv2.erode(newArea, kernel, cv2.MORPH_ERODE)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# img = cv2.Canny(img, 127, 255)
# 画集聚点
for i in range(K):
    img[(center[i, 0]-5): (center[i, 0]+5), (center[i, 1]-5): (center[i, 1]+5)] = (0, 0, 255)
    # img[center[:,0], center[:, 1]] = (0, 0, 255)
newArea = cv2.morphologyEx(newArea, cv2.MORPH_OPEN, kernel)
newArea = cv2.Canny(newArea, 127, 127)

cv2.imshow('Result', img)
cv2.imshow('NewArea', newArea)
cv2.waitKey(0)
cv2.destroyAllWindows()
