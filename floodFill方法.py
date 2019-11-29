import cv2
import numpy as np
import time

"""
log:
1、【实现竖直方向的seed选取---19.11.08】seed点选取的选择算法：若涂色数量小于某个阈值，向上y像素再次判断，
    直到超过阈值
2、【已实现】水平线以上画面置黑，减少干扰（一段时间进行一次判断）
3、不同光照条件下自动选取floodfill明暗阈值（通过直方图峰值？）
4、【实现左右取中值】路径走势计算（边缘走势？十字法？）
5、动作规划（直线曲率？）
"""


def cal_floodFill(image):
    """
    floodFill 计算地面方法
    :param image:Any
    :return: copyImg, threImg
    """
    copyImg = cv2.medianBlur(image, 3)

    # BGR转换为HSV进行处理
    copyImg = cv2.cvtColor(copyImg, cv2.COLOR_BGR2HSV)

    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], dtype=np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列

    # 过高区域不进行处理
    # mask[0:50][:] = 255

    # 计算种子点
    seedThreshold = 20000   # 最少像素值
    timesLimit = 5         # 计算次数限制
    seed = [319, 479]       # 以画面中间最下面的点为起始点 （x, y）
    times = 0               # 循环次数，若超过阈值则返回(None,None)
    seedMoveDistance = int(seed[1] / timesLimit)    # 失败后上升的距离

    while True:
        # floodFill
        cv2.floodFill(copyImg, mask, tuple(seed), (255, 255, 255), (10,255,255), (60,255,255), flags=cv2.FLOODFILL_FIXED_RANGE)

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
    kernel = np.ones((63, 63), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((35, 35), dtype=np.uint8)
    # threImg = cv2.morphologyEx(threImg, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((13, 13), dtype=np.uint8)
    threImg = cv2.morphologyEx(threImg, cv2.MORPH_ERODE, kernel)

    # 色彩空间转换BGR
    copyImg = cv2.cvtColor(copyImg, cv2.COLOR_HSV2BGR)

    return copyImg, threImg




"""
Main
"""
camera = cv2.VideoCapture(0)

while True:
    # 开始计时
    time_start = time.perf_counter()

    # 获取图像
    # ret, src = camera.read()
    src = cv2.imread('./testLib/camera/16.jpg')

    img = cv2.resize(src, (640, 480))
    copyImg, threImg = cal_floodFill(img)

    import roadCal.roadCal as rc
    if threImg is None:
        continue
    threImg = cv2.GaussianBlur(threImg, (53,53), sigmaX=0)
    # line, direct = rc.hough(cv2.Canny(threImg, 100, 127))

    theta = rc.fitRoad_cross(threImg, 20)

    if theta == 0:
        print(f'Straight!')
    else:
        print(f'Turn!')
        print('Theta: ', theta)

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
