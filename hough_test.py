import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.resize(cv2.imread('./testRoad.jpg'), (640, 480))


canny = cv2.Canny(img, 50, 200)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000 *(a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        # 显示特定角度的直线
        if not 1.4 < theta < 2:
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            print(theta)

cv2.imshow('Canny', canny)
cv2.imshow('Img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
