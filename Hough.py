import cv2
import numpy as np
import math
import time


def blur(src):
    """
    模糊处理
    :param src:
    :return: 模糊图像
    """
    __blur = cv2.medianBlur(src, 3)             # 中值滤波
    # blur = cv2.GaussianBlur(img, (3, 3), 0)   # 高斯滤波
    return __blur


def edge(src):
    """
    边界处理
    :param src:
    :return: 边界图像
    """
    __edge = cv2.Canny(src, 150, 220)  # canny边缘检测
    # edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)     # 自适应阈值
    return __edge


def hough(imgEdge, src=None):
    """
    Hough处理
    :param imgEdge: 边界图像
    :param src: 画布
    :return:画线的图像，拟合角度
    """
    # 若无输入画布则为黑布
    if src is None:
        src = np.zeros(imgEdge.shape, dtype=np.uint8)

    lines = cv2.HoughLines(imgEdge, 1, np.pi / 180 * 2, 110)
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
# img = cv2.resize(cv2.imread('./testLib/Low/3.jpg', cv2.IMREAD_GRAYSCALE), (640, 480))
camera = cv2.VideoCapture(1)

while True:
    # 开始计时
    time_start = time.perf_counter()

    # 图像初步处理
    ret, frame = camera.read(cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (640, 480))

    # 开始加工
    img_blur = blur(img)                            # 模糊
    img_edge = edge(img_blur)                       # 边缘识别
    img_hough, img_direct = hough(img_edge, img)    # 直线拟合

    # 显示结果
    cv2.imshow('Step1.Raw', img)
    cv2.imshow('Step2.Blur', img_blur)
    cv2.imshow('Step3.Edge', img_edge)
    if img_hough is not None:
        cv2.imshow('Step4.Hough', img_hough)
        print(f'direct:{img_direct}')

    # 显示处理时间
    time_elapsed = (time.perf_counter() - time_start)
    print("Time used:", time_elapsed)

    # Esc退出
    keyAction = cv2.waitKey(50)    # 延时50ms
    if keyAction == 27:  # Esc
        cv2.destroyAllWindows()
        break
