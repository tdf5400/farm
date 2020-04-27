import cv2
import numpy as np

src = cv2.imread("./0.jpg", cv2.IMREAD_GRAYSCALE)
src = cv2.resize(src, (320, 240))

# 颜色反转
h, w = src.shape[:2]
for i in range(h):
    for j in range(w):
        src[i, j] = 255-src[i, j]

cv2.imshow("raw", src)

kernel = np.ones((15, 15), dtype=np.uint8)
erode = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
dilate = cv2.morphologyEx(src, cv2.MORPH_DILATE, kernel)

cv2.imshow("erode", erode)
cv2.imshow("dilate", dilate)

cv2.waitKey(0)
cv2.destroyAllWindows()
