import cv2
import numpy as np
import math
import time


def cal_blur(src):
    output = src
    output = cv2.medianBlur(output, 7)  # 中值滤波
    # output = cv2.GaussianBlur(output, (5, 5), 0)   # 高斯滤波
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
    criteria = (cv2.TERM_CRITERIA_EPS, 5, 5.0)    # 标准
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


def cal_morphology(src):
    """
    形态学处理
    :param src: 待处理图像
    :return: output 处理结果
    """
    output = src
    kernel = np.ones((25, 25), dtype=np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)  # 开运算
    kernel = np.ones((7, 7), dtype=np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)  # 闭运算

    kernel = np.ones((5, 5), dtype=np.uint8)
    output = cv2.GaussianBlur(output, (7, 7), sigmaX=0, sigmaY=5)  # 高斯模糊平滑边界
    # kernel = np.ones((75, 75), dtype=np.uint8)
    # output = cv2.erode(output, kernel, cv2.MORPH_ERODE)   # 腐蚀
    return output


def cal_edge(src):
    """
    边界计算
    :param src:
    :return: 边缘图像
    """
    # # newArea = cv2.morphologyEx(newArea, cv2.MORPH_OPEN, kernel)
    # # newArea = cv2.Canny(newArea, 127, 127)
    __edge = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)     # 自适应阈值
    # canny = cv2.Canny(newArea, 10, 170)   # canny
    return __edge


def cal_hough(imgEdge, src=None):
    """
    Hough处理
    :param imgEdge: 边界图像
    :param src: 画布
    :return:画线的图像，拟合角度
    """
    # 若无输入画布则为黑布
    if src is None:
        src = np.zeros(imgEdge.shape, dtype=np.uint8)

    lines = cv2.HoughLines(edge, 1, np.pi / 180 * 1, 150)
    if lines is not None:
        __ground = np.zeros(imgEdge.shape, dtype=np.uint8)
        line_direct_theta = 0  # 记录角度
        line_direct_times = 0  # 取平均用
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # y0=kx0+b    rho=
            pt1 = (int(x0 + 600 * (-b)), int(y0 + 600 * a))
            pt2 = (int(x0 - 600 * (-b)), int(y0 - 600 * a))

            if not theta == 0:
                __lineTheta = math.atan(-1 / math.tan(theta)) / np.pi * 180  # 直线角度
            else:
                __lineTheta = 0
            # 显示特定角度的直线
            if not -15 < __lineTheta < 15 or 75 < line_direct_theta < 105:
                cv2.line(src, pt1, pt2, 255, 3, cv2.LINE_AA)
                line_direct_theta += theta
                line_direct_times += 1
                # 打印每条线的信息
                # print(rho, __lineTheta)
        # cv2.imshow('Setp4.Hough', src)
        # 计算指引线
        if line_direct_times is not 0:
            line_direct_theta = line_direct_theta / line_direct_times
            line_direct_output = line_direct_theta / np.pi * 180
            # print(f'direct:{line_direct_output}')
            return src, line_direct_output
    return None, None







"""
Main Function
"""
camera = cv2.VideoCapture(1)

while True:
    # 开始计时
    time_start = time.perf_counter()

    # 图像初步处理
    # frame = cv2.resize(cv2.imread('./testLib/Low/3.jpg'), (640, 480))
    ret, frame = camera.read()
    img = cv2.resize(frame, (640, 480))

    # 开始计算
    blur = cal_blur(img)                            # 模糊
    kmeans = cal_kmeans(blur)                       # K-Means计算
    morphology = kmeans # 腐蚀、平滑
    #cal_morphology(kmeans)             # 形态学运算
    edge = cal_edge(morphology)                     # 边缘计算
    img_hough, img_direct = cal_hough(edge, img)    # 直线拟合

    # 结果显示
    # cv2.imshow('kmeans', kmeans)
    cv2.imshow('Step1.Blur', blur)
    # cv2.imshow('Step2.Morphology', morphology)
    cv2.imshow('Step3.Edge', edge)
    if img_hough is not None:
        cv2.imshow('Step4.Hough', img_hough)
        print(f'direct:{img_direct}')

    # 显示处理时间
    time_elapsed = (time.perf_counter() - time_start)
    print("Time used:", time_elapsed, '\nFre:', (1/time_elapsed))

    # Esc退出
    keyAction = cv2.waitKey(1)  # 延时1ms
    if keyAction == 27:  # Esc
        cv2.destroyAllWindows()
        break
