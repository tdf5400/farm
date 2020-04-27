# -*- coding : utf-8 -*-
import cv2
import numpy as np
import time
import roadCal.roadCal as rc


def floodFill(image):
    """
    floodFill 计算地面方法
    :param image:Any
    :return: copyImg, threImg
    """
    # __FILLCOLOR - floodfill时填充的颜色
    __FILLCOLOR = (255, 255, 255)  # 绿色

    # 预处理
    copyImg = cv2.medianBlur(image, 3)

    # BGR转换为HSV进行处理
    copyImg = cv2.cvtColor(copyImg, cv2.COLOR_BGR2HSV)

    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], dtype=np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列

    # 过高区域不进行处理
    # mask[0:50][:] = 255

    # 计算种子点
    # seedThreshold = int(h * w / 7.5)  # 20000   # 最少像素值
    # timesLimit = 5  # 计算次数限制
    seed = [int(w / 2) - 1, h - 1]  # 以画面中间最下面的点为起始点 （x, y）
    # times = 0  # 循环次数，若超过阈值则返回(None,None)
    # seedMoveDistance = int(seed[1] / timesLimit)  # 失败后上升的距离

    while True:
        # floodFill
        # --------------------------------19.12.21 阈值数据备份--------------------------------
        # cv2.floodFill(copyImg, mask, tuple(seed), (255, 255, 255), (20, 100, 255), (40, 150, 255),
        #               flags=cv2.FLOODFILL_FIXED_RANGE)
        # ------------------------------------------------------------------------------------
        cv2.floodFill(copyImg, mask, tuple(seed), (255, 255, 255), (20, 100, 255), (40, 150, 255),
                      flags=cv2.FLOODFILL_FIXED_RANGE)

        # 二值化并统计渲染数量
        threImg = cv2.inRange(copyImg, copyImg[seed[1], seed[0]], copyImg[seed[1], seed[0]])  # 将与种子点一样被染色的点划出来
        break

    kernel = np.ones((65, 65), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_CLOSE, kernel)

    # kernel = np.ones((35, 35), dtype=np.uint8)
    # threImg = cv2.morphologyEx(threImg, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((7, 7), dtype=np.uint8)
    # threImg = cv2.morphologyEx(threImg, cv2.MORPH_ERODE, kernel)

    # 色彩空间转换BGR
    copyImg = cv2.cvtColor(copyImg, cv2.COLOR_HSV2BGR)
    return copyImg, threImg

def HSV_threshold(src, block, threshold):
    """
    HSV阈值法
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
    # cv2.imshow('Demo', imgThre)
    output = imgThre
    # kernel = np.ones((13, 13), dtype=np.uint8)
    # output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((27, 27), dtype=np.uint8)
    # output = cv2.morphologyEx(output, cv2.MORPH_ELLIPSE, kernel)
    return output

def cal_kmeans(src):
    """
    K-Means 处理
    :param src:
    :return:
    """
    height, wide = src.shape[0:2]   # 读取图像形状

    iData = src.reshape((-1, 3))    # 将三维数组变成二维数组
    iData = np.float32(iData)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS, 8, 5.0)    # 标准
    K = 3   # 聚类数量
    ret, label, center = cv2.kmeans(iData, K, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    oData = np.resize(label.ravel(), (height, wide))  # 录入k-means结果

    # 获取地面区域组号
    if oData is not None:
        value = 0
        counter = 0
        simple_x = int(wide * 0.8)
        simple_y = int(height * 0.8)

        simple = np.resize(oData, (simple_x, simple_y))    # 缩小计算样本，减少计算时间
        # # 判定条件：区域大的为地面
        # for i in range(K):
        #     temp = np.sum(simple == i)
        #     if counter < temp:
        #         counter = temp
        #         value = i

        # 判定条件：底部区域平均值计算
        value = 0.0
        temp = oData[(simple_y-30):(simple_y-10), (int(simple_x*0.5)-10):(int(simple_x*0.5)+10)]
        for i in np.resize(temp, 1):
            value += i/400
        value = int(value * 400)

        # 给目标区域上色
        __area = np.zeros((height, wide), dtype=np.uint8)  # 存储包含路面的新区域
        for i in range(height):
            for j in range(wide):
                if oData.item((i, j)) == value:
                    __area.itemset((i, j), 255)
        return __area

def main():
    src = cv2.imread("../../testLib/farm/1.jpg")
    src = cv2.resize(src, (640, 480))
    cv2.imshow("raw", src)

    if src is None:
        return

    # 固定种子点floodfill
    # floodfill_img, floodfill_thre = floodFill(src)
    # cv2.imshow("floodfill_img", floodfill_thre)

    # 正式版floodfill
    floodfill_img, floodfill_thre = rc.cal_floodFill(src, (20, 100, 255), (40, 150, 255))
    cv2.imshow("floodfill_img_New", floodfill_thre)

    # HSVthre_thre = HSV_threshold(cv2.cvtColor(src, cv2.COLOR_RGB2HSV), (300, 440, 340, 480), (150, 1000, 1000))
    # cv2.imshow("HSVthre_img", HSVthre_thre)
    #
    # kmeans_thre = cal_kmeans(src)
    # cv2.imshow("kmeans_img", kmeans_thre)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
