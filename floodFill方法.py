import cv2
import numpy as np
import time

"""
log:
1、seed点选取的选择算法：若涂色数量小于某个阈值，向上y像素再次判断，
    直到超过阈值
2、水平线以上画面置黑，减少干扰（一段时间进行一次判断）
"""


def fill_color_demo(image):
    # floodFill
    seed = (480, 320)   # 以画面中间最下面的点为基准
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列
    # 为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    cv2.floodFill(copyImg, mask, seed, (255, 255, 255), (55,50,50), (55,45,45), flags=cv2.FLOODFILL_FIXED_RANGE)

    # 形态学运算
    threImg = cv2.inRange(copyImg, copyImg[320, 480], copyImg[320, 480])    # 二值化
    kernel = np.ones((7, 7), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((15, 15), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((29, 29), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_ERODE, kernel)

    return copyImg, threImg

"""
Main
"""
camera = cv2.VideoCapture(0)

while True:
    # 开始计时
    time_start = time.perf_counter()

    ret, src = camera.read()
    # src = cv2.imread('./testLib/Low/8.jpg')
    img = cv2.resize(src, (640, 480))

    copyImg, threImg = fill_color_demo(img)
    cv2.imshow('input_image', img)
    cv2.imshow('floodFill', copyImg)
    cv2.imshow('Threshold', threImg)

    # 显示处理时间
    time_elapsed = (time.perf_counter() - time_start)
    print("Time used:", time_elapsed, '\nFre:', (1 / time_elapsed))

    # Esc退出
    keyAction = cv2.waitKey(1)  # 延时1ms
    if keyAction == 27:  # Esc
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
