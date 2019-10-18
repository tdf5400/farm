import cv2
import numpy as np

img_raw = cv2.imread('./0_threshold.png')

kernel_0 = np.ones((3, 3), dtype=np.uint8)
kernel_1 = np.ones((36, 36), dtype=np.uint8)


img = cv2.erode(img_raw, kernel_0)
img = cv2.erode(img, kernel_0)
img = cv2.dilate(img, kernel_1)
cv2.imshow('Close', img)


# 直线拟合部分
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarfchy = cv2.findContours(thresh, cv2.CONTOURS_MATCH_I1, 2)
cnt = contours[0]
M = cv2.moments(cnt)

rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
img = cv2.line(img, (cols - 1, righty), (0, lefty), (0,255,0), 2)
cv2.imshow('Demo', img)

# cv2.fitLine()

cv2.waitKey(0)
cv2.destroyAllWindows()
