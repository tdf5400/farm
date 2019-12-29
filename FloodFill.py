import cv2
import numpy as np
import time
import roadCal.roadCal as rc

"""
log:
1、【实现竖直方向的seed选取---19.11.08】seed点选取的选择算法：若涂色数量小于某个阈值，向上y像素再次判断，
    直到超过阈值
2、【已实现】水平线以上画面置黑，减少干扰（一段时间进行一次判断）
3、不同光照条件下自动选取floodfill明暗阈值（通过直方图峰值？）
4、【实现左右取中值】路径走势计算（边缘走势？十字法？）
5、【通过十字法实现】动作规划（直线曲率？）
6、【已实现】若道路两侧是开放的，做出相应判断
7、【fitRoad_middle函数拟合直线部分未完成】直线状态下的姿态纠正
8、有障碍物存在的直线规划
9、摄像头被遮挡判断
"""



# """
# Main
# """
# camera = cv2.VideoCapture(0)
#
# while True:
#     # 开始计时
#     time_start = time.perf_counter()
#
#     # 获取图像
#     # ret, src = camera.read()
#     # src = cv2.imread('./testLib/camera/17.jpg')
#     src = cv2.imread('C:/Users/tdf54/Desktop/TestImg/02.jpg')
#     if src is None:         # 图像存在性判断
#         print(f'No Image!')
#         continue
#
#     img = cv2.resize(src, (640, 480))       # 分辨率重定义
#     copyImg, threImg = cal_floodFill(img)   # FloodFill计算
#     if threImg is None:                     # 取色失败则进入下一帧
#         print(f'FloodFill Error!')
#         continue
#
#     threImg = cv2.GaussianBlur(threImg, (53,53), sigmaX=0)
#     # line, direct = rc.hough(cv2.Canny(threImg, 100, 127))
#
#     state, theta = rc.fitRoad_cross(threImg, 20)
#
#     if state == rc.FIT_CROSS_STRAIGHT:
#         print(f'Straight!', end='\t')
#         print('Theta:', theta)
#     elif state == rc.FIT_CROSS_TRUN:
#         print(f'Turn!', end='\t')
#         print('Theta: ', theta)
#     elif state == rc.FIT_CROSS_OUT:
#         print(f'Out of Road!')
#     else:
#         print(f'Error!', end='\t')
#         print('Info:', theta)
#
#     cv2.imshow('floodFill', copyImg)
#     cv2.imshow('Threshold', threImg)
#
#     # 显示处理时间
#     time_elapsed = (time.perf_counter() - time_start)
#     print("Time used:", time_elapsed, '\nFre:', (1 / time_elapsed))
#
#     # Esc退出
#     keyAction = cv2.waitKey(1)  # 延时1ms
#     if keyAction == 27:  # Esc
#         cv2.destroyAllWindows()
#         break
#
# cv2.destroyAllWindows()
