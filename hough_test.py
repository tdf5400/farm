import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.resize(cv2.imread('./testRoad_1.jpg', cv2.IMREAD_GRAYSCALE), (640, 480))


canny = cv2.Canny(img, 50, 200)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 150)

if lines is not None:
    for i in range(0, len(lines)):
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
        # 显示特定角度的直线
        if not -15 < lineTheta < 15:
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            print(rho, lineTheta)

cv2.imshow('Canny', canny)
cv2.imshow('Img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
