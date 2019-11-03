"""
19.10.18 使用运算后的结果并进行线性回归，结果不理想
"""

import cv2
import numpy as np

img_raw = cv2.imread('./0_threshold.png')


kernel_0 = np.ones((7, 7), dtype=np.uint8)
kernel_1 = np.ones((25, 25), dtype=np.uint8)

img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# img = cv2.equalizeHist(img)
img = cv2.erode(img, kernel_0)
img = cv2.dilate(img, kernel_1)
cv2.imshow('Close', img)


# 直线拟合部分
contours, hierarfchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
M = cv2.moments(cnt)

rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L1, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
img = cv2.line(img_raw, (cols - 1, righty), (0, lefty), (0,255,0), 2)
cv2.imshow('Demo', img_raw)


cv2.waitKey(0)
cv2.destroyAllWindows()
