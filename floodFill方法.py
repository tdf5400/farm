import cv2
import numpy as np
import time

"""
log:
1、【实现竖直方向的seed选取---19.11.08】seed点选取的选择算法：若涂色数量小于某个阈值，向上y像素再次判断，
    直到超过阈值
2、水平线以上画面置黑，减少干扰（一段时间进行一次判断）
"""


def cal_floodFill(image):
    """
    floodFill 计算地面方法
    :param image:Any
    :return: copyImg, threImg
    """
    copyImg = cv2.medianBlur(image, 3)
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], dtype=np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列

    # 计算种子点
    seedThreshold = 20000   # 最少像素值
    timesLimit = 5         # 计算次数限制
    seed = [319, 479]       # 以画面中间最下面的点为起始点 （x, y）
    times = 0               # 循环次数，若超过阈值则返回(None,None)
    seedMoveDistance = int(seed[1] / timesLimit)    # 失败后上升的距离

    while True:
        # floodFill
        cv2.floodFill(copyImg, mask, tuple(seed), (255, 255, 255), (55,50,50), (65,60,60), flags=cv2.FLOODFILL_FIXED_RANGE)

        # 二值化并统计渲染数量
        threImg = cv2.inRange(copyImg, copyImg[seed[1], seed[0]], copyImg[seed[1], seed[0]])    # 将与种子点一样变成白色的点划出来
        threCounter = np.sum(threImg == 255)    # 统计出现的数量

        # 退出的判定
        if threCounter >= seedThreshold:
            break
        else:
            times += 1
            if times < timesLimit:
                seed[1] -= seedMoveDistance   # seed上移

            else:
                return None, None

    # 形态学运算
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

    # 获取图像
    ret, src = camera.read()
    # src = cv2.imread('./testLib/Low/8.jpg')
    img = cv2.resize(src, (640, 480))

    copyImg, threImg = cal_floodFill(img)
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
