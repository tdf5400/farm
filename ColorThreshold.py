# _*_ coding: utf-8 _*_
import numpy as np
import cv2


def get_simple(src, block, threshold):
    """
    获取样本区域
    :param src: hsv图像
    :param block: 区域, tuple(x0, y0, x1, y1) 从左上角开始
    :param threshold: 阈值  tuple(H, S, V)
    :return: thresholdImg: 阈值分割结果
    """
    x0, y0, x1, y1 = block[0:4]

    # 小的值放在前面
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    #  计算相关信息
    width = x1 - x0
    height = y1 - y0
    ave_times = int(width * height)
    ave_h = 0.0  # h平均值
    threshold_h = int(threshold[0]/2)  # h阈值
    threshold_s = int(threshold[1]/2)  # s阈值
    threshold_v = int(threshold[2]/2)  # v阈值

    # H 均值
    for i in range(0, width):
        for j in range(0, height):
            ave_h += src.item((y0 + j), (x0 + i), 0) / ave_times    # [y][x][deep]
    ave_h = int(ave_h)
    imgThre = cv2.inRange(src, ((ave_h - threshold_h), (127-threshold_s), (127-threshold_v)),
                                ((ave_h + threshold_h), (127+threshold_s), (127+threshold_v)))
    cv2.imshow('Demo', imgThre)
    output = imgThre
    kernel = np.ones((13, 13), dtype=np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((27, 27), dtype=np.uint8)
    # output = cv2.morphologyEx(output, cv2.MORPH_ELLIPSE, kernel)
    return output



frame = cv2.imread('./testLib/Low/0.jpg')
frame = cv2.resize(frame, (640, 480))

# 中值滤波
blur = cv2.medianBlur(frame, 23)
blur = get_simple(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), (300, 440, 340, 480), (5, 255, 255))
cv2.imshow('Blur', blur)



cv2.waitKey(0)
cv2.destroyAllWindows()
