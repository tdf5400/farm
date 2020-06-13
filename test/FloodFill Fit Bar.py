# 阈值调整脚本

import cv2
import roadCal.roadCal as rc

CAMERA_FLAG = 0  # 是否使用摄像头

# 全局变量
__img = 0
__camera = 0
path = "../testLib/banana/2.jpg"     # 照片路径

def __refresh(x):
    global __img, __camera

    # 摄像头
    if CAMERA_FLAG:
        ret, src = __camera.read()
        if src is None:  # 判断图像存在性
            print("图像不存在！")
        else:
            __img = cv2.resize(src, (640, 480))  # 分辨率重定义
            # __img = cv2.GaussianBlur(__img, (53, 53), sigmaX=0)

    __h = cv2.getTrackbarPos('loDirr_H', 'Trackbar')
    __s = cv2.getTrackbarPos('loDirr_S', 'Trackbar')
    __v = cv2.getTrackbarPos('loDirr_V', 'Trackbar')
    loDirr = (__h, __s, __v)
    __h = cv2.getTrackbarPos('upDirr_H', 'Trackbar')
    __s = cv2.getTrackbarPos('upDirr_S', 'Trackbar')
    __v = cv2.getTrackbarPos('upDirr_V', 'Trackbar')
    upDirr = (__h, __s, __v)

    mask_wide = cv2.getTrackbarPos('mask_wide', 'Trackbar')
    print(f"Parameter: {loDirr}, {upDirr}, mask_wide={mask_wide}")

    copyImg, threImg = rc.cal_floodFill(__img, loDirr, upDirr, mask_wide=mask_wide)  # FloodFill计算
    if threImg is None:  # 取色失败则进入下一帧
        print("计算失败！")
        try:
            cv2.destroyWindow('Demo')
        except Exception:
            pass
    else:
        cv2.imshow('Demo', copyImg)


def __main():
    global __img, __camera
    print("启动floodfill阈值调试程序！\r\n")
    if CAMERA_FLAG:
        print("使用摄像头！")
        __camera = cv2.VideoCapture(0)
        ret, src = __camera.read()
    else:
        print("使用内置图片！")
        # src = cv2.imread('./testLib/camera/36.jpg')
        src = cv2.imread(path)

    if src is None:  # 判断图像存在性
        print("图像不存在！")
    else:
        __img = cv2.resize(src, (640, 480))  # 分辨率重定义
        # __img = cv2.GaussianBlur(__img, (53, 53), sigmaX=0)

        __refresh(None)



if __name__ == "__main__":
    # 创建调节棒
    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('loDirr_H', 'Trackbar', 100, 255, __refresh)
    cv2.createTrackbar('loDirr_S', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('loDirr_V', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('upDirr_H', 'Trackbar', 100, 255, __refresh)
    cv2.createTrackbar('upDirr_S', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('upDirr_V', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('mask_wide', 'Trackbar', 0, 320, __refresh)

    __main()


    while(True):
        # Esc退出
        keyAction = cv2.waitKey(1)  # 延时1ms
        if keyAction == 27:  # Esc
            cv2.destroyAllWindows()
            break

