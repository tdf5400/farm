# -*- coding : utf-8 -*-
import cv2
import numpy as np
import time
import roadCal.roadCal as rc
import Serial

SERIAL_SWITCH = 1   # 串口控制开关


def main():
    camera = cv2.VideoCapture(0)

    while True:
        # 等待串口指令
        reg = None
        if SERIAL_SWITCH:
            Serial.serialInit("COM3", 115200, 5)
            while reg is None:
                # Esc退出
                keyAction = cv2.waitKey(100)  # 延时100ms
                if keyAction == 27:  # Esc
                    cv2.destroyAllWindows()
                    return -1

                reg = Serial.readReg()



        while True:
            # 开始计时
            time_start = time.perf_counter()

            # 获取图像
            # ret, src = camera.read()
            src = cv2.imread('./testLib/camera/17.jpg')
            if src is None:  # 判断图像存在性
                print(f'[console]No Image!')
                continue

            img = cv2.resize(src, (640, 480))  # 分辨率重定义
            copyImg, threImg = rc.cal_floodFill(img)  # FloodFill计算
            if threImg is None:  # 取色失败则进入下一帧
                print(f'[console]FloodFill Error!')
                continue

            threImg = cv2.GaussianBlur(threImg, (53, 53), sigmaX=0)
            # line, direct = rc.hough(cv2.Canny(threImg, 100, 127))

            state, staInfo = rc.fitRoad_cross(threImg, 20)

            if state == rc.FIT_CROSS_STRAIGHT:
                print(f'[console]Straight!', end='\t')
                print('Theta:', staInfo)
            elif state == rc.FIT_CROSS_TRUN:
                print(f'[console]Turn!', end='\t')
                print('Theta: ', staInfo)
            elif state == rc.FIT_CROSS_OUT:
                print(f'[console]Out of Road!')
            else:
                print(f'[console]Error!', end='\t')
                print('Info:', staInfo)

            cv2.imshow('floodFill', copyImg)
            cv2.imshow('Threshold', threImg)

            # 显示处理时间
            time_elapsed = (time.perf_counter() - time_start)
            print(f'[console]Used:{time_elapsed}')
            print(f'[console]Fre:{(1 / time_elapsed)}')

            # 串口输出
            if reg == Serial.ACQUIRE_STA:
                Serial.sendData(state)
                break
            elif reg == Serial.ACQUIRE_BOTH:
                Serial.sendData(state, int(staInfo*1000))   # 由于状态量为浮点数，所以放大后再发送
                break

            # Esc退出
            keyAction = cv2.waitKey(1)  # 延时1ms
            if keyAction == 27:  # Esc
                cv2.destroyAllWindows()
                return -1





"""
Main
"""
main()
