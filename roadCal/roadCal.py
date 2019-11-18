import cv2
import numpy as np
import math

"""
路径计算方法整合
"""
def A():
    print(111)

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

        lines = cv2.HoughLines(imgEdge, 1, np.pi / 180 * 1, 50)
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

