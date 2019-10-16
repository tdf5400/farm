import cv2
import numpy as np


def refresh_img(x=0):
    """
    刷新图像
    """
    x = x
    global img_raw, color_add, threshold_value
    __img = img_raw
    wide, high = __img.shape[0], __img.shape[1]
    # 获取滑动条位置
    color_add = {'R': 0, 'G': 0, 'B': 0}
    threshold_value = cv2.getTrackbarPos('threshold', 'Bar')
    color_add['R'] = cv2.getTrackbarPos('R', 'Bar') - 255
    color_add['G'] = cv2.getTrackbarPos('G', 'Bar') - 255
    color_add['B'] = cv2.getTrackbarPos('B', 'Bar') - 255

    # 分割图层
    (b, g, r) = cv2.split(__img)
    add_b = np.zeros((wide, high, 1), dtype=np.uint8)
    add_g = np.zeros((wide, high, 1), dtype=np.uint8)
    add_r = np.zeros((wide, high, 1), dtype=np.uint8)
    add_r[:][:] = abs(color_add['R'])
    add_g[:][:] = abs(color_add['G'])
    add_b[:][:] = abs(color_add['B'])

    # 单独操作通道
    if color_add['R'] > 0:
        r = cv2.add(r, add_r)
    else:
        r = cv2.subtract(r, add_r)

    if color_add['G'] > 0:
        g = cv2.add(g, add_g)
    else:
        g = cv2.subtract(g, add_g)

    if color_add['B'] > 0:
        b = cv2.add(b, add_b)
    else:
        b = cv2.subtract(b, add_b)

    # 合并通道
    __img = cv2.merge([b, g, r])
    cv2.imshow('img', __img)
    # 二值化处理
    __img_gray = cv2.cvtColor(__img, cv2.COLOR_RGB2GRAY)
    __img_threshold = cv2.threshold(__img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('Changed', __img_threshold)


img_raw = cv2.imread('./0.jpg')
refresh_img()  # 刷新图像

# 调节窗口
cv2.namedWindow('Bar')
cv2.createTrackbar('threshold', 'Bar', 0, 255, refresh_img)
cv2.createTrackbar('R', 'Bar', 255, 510, refresh_img)
cv2.createTrackbar('G', 'Bar', 255, 510, refresh_img)
cv2.createTrackbar('B', 'Bar', 255, 510, refresh_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
