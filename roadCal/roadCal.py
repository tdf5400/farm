import cv2
import numpy as np
import math

"""
路径计算方法整合
"""

def fitRoad_cross(imgThre, threshold):
    """
    计算路径（十字法）
    :param imgThre: 二值化图像
    :param threshold:  阈值（检测到顶部后，下移距离）
    :return:曲率数据
    """
    output = imgThre.copy()
    height, width = output.shape[0:2]
    theta = 0   # 中线斜率，负-左边，正-右边

    # 检测边缘
    # output = cv2.Canny(output, 1, 120)    # 结果不明显
    output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 0)

    # 从中点向上前进
    middlePoint = int(width/2)
    edge = [0, width]   # [左边缘， 右边缘]
    for i in range(height-1, -1, -1):
        if output.item((i, middlePoint)) == 255:    # 向上前进遇到边缘
            startPoint = i + threshold
            # 向左检测
            for j in range(middlePoint, -1, -1):
                if output.item((startPoint, j)) == 255:
                    edge[0] = j
                    break
            # 向右检测
            for j in range(middlePoint, width):
                if output.item((startPoint, j)) == 255:
                    edge[1] = j
                    break

            # 判断左右开口,取离中点近的值
            judge = 0   # 0-左，1-右
            if abs(edge[0] - middlePoint) < abs(edge[1] - middlePoint):
                judge = 0
            else:
                judge = 1
            # 计算曲率
            # K = 1/r = (4 * Sabc) / (a*b*c)
            # Sabc = [(a+b+c)(a+b)(a+c)(b+c)]**0.5
            A = [height - 1, middlePoint]   # 曲线起点
            B = [startPoint, edge[judge]]   # 左/右交叉点
            C = (i, middlePoint)            # 顶点

            # 获取曲线起点【A点】（从两边开始）
            for j in range(0 if judge==0 else (width - 1), middlePoint, 1 if judge==0 else -1):
                if output.item(height-1, j) == 255:
                    A[1] = j
                    break
            else:   # 底层找不到则上浮
                for j in range(height, height-threshold, -1):
                    for k in range(0 if judge==0 else (width - 1), middlePoint, 1 if judge==0 else -1):
                        if output.item(height - 1, j) == 255:
                            A = (j, k)
                            j = -1  # 退出双重循环
                            break
                    if j == -1:
                        break

            # 边界长度
            a = ((abs(B[0] - C[0]))**2 + (abs(B[1] - C[1]))**2)**0.5
            b = ((abs(A[0] - C[0]))**2 + (abs(A[1] - C[1]))**2)**0.5
            c = ((abs(B[0] - A[0]))**2 + (abs(B[1] - A[1]))**2)**0.5

            Sabc = ((a+b+c)*(a+b)*(a+c)*(b+c))**0.5 # 三角形abc面积
            K = (4*Sabc) / (a*b*c)  # 曲率

            theta = np.arctan(K)/np.pi*180
            theta /= 2  # 范围压缩到0-90
            theta *= -1 if judge == 1 else 1  # 左边空-负 右边空-正

            break
        else:
            theta = 0

    return theta



def fitRoad_middle(imgThre):
    """
    计算路径（以每行）
    :param imgThre: 路径的二值化图像
    :return: 添加点的图像
    """
    output = imgThre.copy()
    height, width = output.shape[0:2]

    for i in range(height):
        edge_L = 0
        edge_R = width - 1

        # 查找左边界
        for j in range(int(width/2), -1, -1):
            if output.item((i, j)) == 0:# and output.item((i, j+1)) == 255:
                edge_L = j  # 记录边界
                break

        # 查找右边界
        for j in range(int(width/2), width, 1):
            if output.item((i, j)) == 0:# and output.item((i, j - 1)) == 255:
                edge_R = j  # 记录边界
                break

        # 计算中心
        middle = int((edge_L + edge_R)/2)

        output.itemset((i, middle), 0)


    return output


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

