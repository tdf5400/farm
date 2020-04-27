import cv2
import numpy as np
import math

"""
路径计算方法整合
20.02.16 使用B、C点的直线斜率结合曲率计算负反馈进行斜率输出
         更改斜率判断规则，k>0-左边，k<0-右边
20.02.17 出田垄时返回的数据为“图像中田地占画面高的比例”
"""

## 状态标志
FIT_CROSS_STRAIGHT = 0  # 直道
FIT_CROSS_TRUN = 1      # 弯道
FIT_CROSS_OUT = 2       # 出林道
FIT_CROSS_ERR = -1      # 错误

## 设置参数
OUT_THRESHOLD = 0.65    # 出田垄进度计算-阈值


def fitRoad_Out_Conf(imgThre, threshold):
    """
    计算出田垄后，图像中田地占画面高的比例
    :param imgThre: 二值化图像
    :param threshold: 判定条件（百分数，0~1）
    （一行中有threshold%的值<127-->>道路）
    :return: 数值正确 - 道路/画面高度
             数值错误 - 0
    """
    # 合法性判断
    if type(imgThre).__name__ != 'ndarray' or (threshold<0 or threshold>1):
        return 0

    threshold *= 100
    copyImg = imgThre.copy()
    height, width = copyImg.shape[0:2]

    step = height/100  # 步长
    times = 0               # 道路次数
    errTimes = 0            # 3次判定失败则退出
    # 行遍历
    data = copyImg[(height-1)::-1, ]  # 切片
    for i in range(100):
        temp = data[int(i*step)]

        if np.percentile(temp, threshold) < 127:
            times += 1
            errTimes = 0    # 清空错误次数
        else:
            errTimes += 1
            if errTimes >= 3:
                break

    if times == 0:
        return 0
    else:
        return times/100



def fitRoad_cross(imgThre, threshold):
    """
    计算路径（十字法）
    :param imgThre: 二值化图像
    :param threshold:  阈值（检测到顶部后，下移距离）
    :return: 状态, 数据
                    FIT_CROSS_STRAIGHT, 斜率（左-负 右-正）
                    FIT_CROSS_TRUN,     曲率（左-负 右-正）
                    FIT_CROSS_OUT,      0
                    FIT_CROSS_ERR,      错误号
    """
    # 参数设定
    # W_cir - K值计算中圆曲率权值，K=K_BC-W_cir*K_cir
    W_cir = 0.2
    # Out_thre - 出田垄判断阈值，顶部有跳变时，两边跳变点
    #            与边缘距离均小于Out_thre判断为出田垄
    Out_thre = 3

    copyImg = imgThre.copy()
    height, width = copyImg.shape[0:2]

    # 检测边缘
    edgeImg = cv2.adaptiveThreshold(copyImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 0)

    # 从中点向上前进
    middlePoint = int(width / 2)
    edge = [0, width]  # [左边缘， 右边缘]
    for i in range(height - 1, -1, -1):
        if edgeImg.item((i, middlePoint)) == 255:  # 向上前进遇到边缘
            startPoint = i + threshold if (i + threshold) < (height - 1) else (height - 2)
            # 向左检测（因为二值化图像经过高斯模糊，所以127也允许）
            for j in range(middlePoint, -1, -1):
                if edgeImg.item((startPoint, j)) == 127 or edgeImg.item((startPoint, j)) == 255:
                    edge[0] = j
                    break
            # 向右检测
            for j in range(middlePoint, width):
                if edgeImg.item((startPoint, j)) == 127 or edgeImg.item((startPoint, j)) == 255:
                    edge[1] = j
                    break

            ## 判断左右开口,取离中点近的值
            judge = 0  # 0-左，1-右
            distance_l, distance_r = abs(edge[0] - middlePoint), abs(edge[1] - middlePoint)
            # 跳变点均靠近边缘则判断为出田垄
            if (((width/2)-distance_l) < Out_thre) and (((width/2)-distance_r) < Out_thre):
                road_rate = fitRoad_Out_Conf(edgeImg, OUT_THRESHOLD)
                return FIT_CROSS_OUT, road_rate
            # 朝向判断
            if distance_l < distance_r:
                judge = 0
            else:
                judge = 1
            # 计算曲率
            # K = 1/r = (4 * Sabc) / (a*b*c)
            # Sabc = [(a+b+c)(a+b)(a+c)(b+c)]**0.5
            A = [height - 1, middlePoint]  # 曲线起点
            B = [startPoint, edge[judge]]  # 左/右交叉点
            C = (i, middlePoint)  # 顶点

            # 获取曲线起点【A点】（从两边开始）
            for j in range(0 if judge == 0 else (width - 1), middlePoint, 1 if judge == 0 else -1):
                if edgeImg.item(height - 1, j) == 255:
                    A[1] = j
                    break
            else:  # 底层找不到则上浮
                for j in range((height-1), height - threshold, -1):
                    for k in range(middlePoint, width if judge == 0 else -1, 1 if judge == 0 else -1):
                        if edgeImg.item(j, k) == 255:
                            A = (j, k)
                            j = -1  # 退出双重循环
                            break
                    if j == -1:
                        break

            ## 计算BC斜率K_BC（竖直方向为x轴方向,左方向为y轴方向，底层中点为原点O）
            # 坐标系转换
            A = (height - A[0], width / 2 - A[1])
            B = (height - B[0], width / 2 - B[1])
            C = (height - C[0], width / 2 - C[1])
            if A == B == C:     # 数值正确性判断
                return FIT_CROSS_ERR, 0
            # 斜率计算
            K_BC = (C[1]-B[1])/(C[0]-B[0])

            ## 计算K值
            K_AC = (C[1]-A[1])/(C[0]-A[0])  # 计算AC斜率，判断是否在同一直线上
            if K_BC == K_AC:
                K = K_BC
            else:
                # 计算曲线上边界长度
                a = ((abs(B[0] - C[0])) ** 2 + (abs(B[1] - C[1])) ** 2) ** 0.5
                b = ((abs(A[0] - C[0])) ** 2 + (abs(A[1] - C[1])) ** 2) ** 0.5
                c = ((abs(B[0] - A[0])) ** 2 + (abs(B[1] - A[1])) ** 2) ** 0.5

                if a == 0 or b == 0 or c == 0:  # 发生错误
                    return FIT_CROSS_ERR, 0

                Sabc = ((a + b + c) * (a + b) * (a + c) * (b + c)) ** 0.5  # 三角形abc面积
                K_cir = (4 * Sabc) / (a * b * c)  # 曲率
                K_cir = abs(K_cir)  # 曲率取绝对值

                K = K_BC + W_cir*K_cir

            return FIT_CROSS_TRUN, K

    # 遇不到边缘,区分“直道”与“出路口”
    else:
        edge = [0, width]  # [左边缘， 右边缘]
        # 画面底部1/4出向左右两边巡线，若有线则为直道
        # 向左寻线
        for i in range(middlePoint, -1, -1):
            if edgeImg.item(int(height / 4 * 3), i):
                edge[0] = i
        # 向右巡线
        for i in range(middlePoint, width):
            if edgeImg.item(int(height / 4 * 3), i):
                edge[1] = i

        # 两边均无线则判断为出路口
        if edge == [0, width]:
            road_rate = fitRoad_Out_Conf(edgeImg, OUT_THRESHOLD)
            return FIT_CROSS_OUT, road_rate

        # 直道修正
        try:
            lines_theta = fitRoad_middle(copyImg)
            return FIT_CROSS_STRAIGHT, lines_theta
        except IOError as e:  # 错误报告
            return FIT_CROSS_ERR, -3
    return FIT_CROSS_ERR, -1


def fitRoad_middle(imgThre):
    """
    计算路径（以每行）（适用于直道）
    :param imgThre: 路径的二值化图像
    :return: 路线斜率
    """
    copyImg = imgThre.copy()
    height, width = copyImg.shape[0:2]
    road = np.zeros(copyImg.shape, dtype=np.uint8)  # 用于存储路径

    # 扫描
    points = [(0, int(width / 2)), ]  # 计算出来的点,从下到上

    for i in range(height - 2, -1, -1):  # 从底部第二行开始
        edge = [0, width - 1]

        # 从上个点的位置查找左边界
        for j in range(points[(height - 2 - i)][1], -1, -1):
            if copyImg.item((i, j)) == 0:  # and copyImg.item((i, j+1)) == 255:
                edge[0] = j  # 记录边界
                break
        # 从上个点的位置查找右边界
        for j in range(points[(height - 2 - i)][1], width, 1):
            if copyImg.item((i, j)) == 0:  # and copyImg.item((i, j - 1)) == 255:
                edge[1] = j  # 记录边界
                break

        # 计算中心
        middle = int((edge[0] + edge[1]) / 2)
        points.append(((height-1 - i), middle))  # 记录数据

    # 画路径
    for i in points:
        road.itemset((height-1 - i[0], i[1]), 255)
    cv2.imshow('Road', road)

    # 拟合直线
    points = np.uint16(points)

    # 一元线性回归 (yi = a + b * xi)
    x_average = width / 2                   # x平均值
    y_average = sum(points[:][1]) / height  # y平均值
    b_numerator = 0   # 斜率b的分子
    b_denominator = width * (width + 1) * (width*2 + 1) / 6     # 斜率b的分母

    # 计算b的分子
    for i in points:
        b_numerator += (i[0] - x_average) * (i[1] - y_average)

    # 计算b
    b = b_numerator / b_denominator
    K = -b  # K>0-向左,K<0-向右

    return K


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

    lines = cv2.HoughLines(imgEdge, 1, np.pi / 180 * 1, 19)
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
                line_direct_theta += __lineTheta % 180
                line_direct_times += 1
                # 打印每条线的信息
                # print(rho, __lineTheta)

        # 计算指引线
        if line_direct_times is not 0:
            line_direct_theta = line_direct_theta / line_direct_times
            line_direct_output = line_direct_theta
            # print(f'direct:{line_direct_output}')
            return src, line_direct_output
    return None, None

def cal_floodFill(image, loDiff, upDiff):
    """
    floodFill 计算地面方法
    :param image:Any
    :param loDiff:最低阈值（三变量turple，如：(20, 100, 255)）
    :param upDiff:最大阈值（三变量turple，如：(40, 150, 255)）
    :return: copyImg, threImg
    """
    # __FILLCOLOR - floodfill时填充的颜色
    __FILLCOLOR = (255, 255, 255)  # 绿色
    __FURTHERMORPH = 1  # 更精确的形态学运算开关

    # 预处理
    # copyImg = image
    copyImg = cv2.medianBlur(image, 3)
    # copyImg = cv2.GaussianBlur(image, (29, 29), sigmaX=0)

    # BGR转换为HSV进行处理
    copyImg = cv2.cvtColor(copyImg, cv2.COLOR_BGR2HSV)

    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], dtype=np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列

    # 过高区域不进行处理
    # mask[0:50][:] = 255

    # 计算种子点
    seedThreshold = int(h*w/3)#7.5)  # 20000   # 最少像素值
    timesLimit = 5         # 计算次数限制
    seed = [int(w/2)-1, h-1]       # 以画面中间最下面的点为起始点 （x, y）
    times = 0               # 循环次数，若超过阈值则返回(None,None)
    seedMoveDistance = int(seed[1] / timesLimit)    # 失败后上升的距离

    while True:
        # floodFill
        # --------------------------------19.12.21 阈值数据备份--------------------------------
        # cv2.floodFill(copyImg, mask, tuple(seed), (255, 255, 255), (20, 100, 255), (40, 150, 255),
        #               flags=cv2.FLOODFILL_FIXED_RANGE)
        # ------------------------------------------------------------------------------------
        cv2.floodFill(copyImg, mask, tuple(seed), __FILLCOLOR, loDiff, upDiff,
                      flags=cv2.FLOODFILL_FIXED_RANGE)

        # 二值化并统计渲染数量
        threImg = cv2.inRange(copyImg, copyImg[seed[1], seed[0]], copyImg[seed[1], seed[0]])    # 将与种子点一样被染色的点划出来
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
    if __FURTHERMORPH:  # 精确的形态学运算
        # morph_size - morphImg外扩距离(单边)
        # morphImg - 用于避免边缘导致的形态学运算错误
        morph_size = 20
        morphImg = np.zeros((h+morph_size*2, w+morph_size*2), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                morphImg.itemset(morph_size+i, morph_size+j, threImg.item(i, j))
        threImg = morphImg
    kernel = np.ones((65, 65), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_CLOSE, kernel)

    # kernel = np.ones((35, 35), dtype=np.uint8)
    # threImg = cv2.morphologyEx(threImg, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((7, 7), dtype=np.uint8)
    # threImg = cv2.morphologyEx(threImg, cv2.MORPH_ERODE, kernel)
    if __FURTHERMORPH:  # 精确的形态学运算
        # 取原来图像的框
        temp = threImg.copy()
        threImg = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                threImg.itemset(i, j, temp.item(morph_size+i, morph_size+j))

    # 色彩空间转换BGR
    copyImg = cv2.cvtColor(copyImg, cv2.COLOR_HSV2BGR)

    return copyImg, threImg
