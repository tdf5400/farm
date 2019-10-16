import cv2
import numpy as np

color_add = {'R': 0, 'G': 68, 'B': 0}
threshold_value = 0


"""
刷新图像
"""
def refresh_img(x=0):
    x = x
    global img_raw, color_add, threshold_value
    __img = img_raw
    wide, high = __img.shape[0], __img.shape[1]
    # 获取滑动条位置
    # color_add['R'] = cv2.getTrackbarPos('R', 'Bar') - 255
    # color_add['G'] = cv2.getTrackbarPos('G', 'Bar') - 255
    # color_add['B'] = cv2.getTrackbarPos('B', 'Bar') - 255
    # 分割图层
    (b, g, r) = cv2.split(__img)
    add_b = add_g = add_r = np.zeros((wide, high, 1), dtype=np.uint8)
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
    print(__img[0][0])
    cv2.imshow('Demo', __img)
    # # 创建混合图层
    # color = np.zeros((wide, high, 3), dtype=np.int16)
    # temp = np.array([np.uint8(color_add['R']), np.uint8(color_add['G']), np.uint8(color_add['B'])], dtype=np.uint8)
    # color[:][:] = temp
    # # 原图像与混合图层相加
    # __img = cv2.add(__img, color)
    # __img_gray = cv2.cvtColor(__img, cv2.COLOR_RGB2GRAY)
    # __img_threshold = __img_gray #cv2.threshold(__img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('Demo', __img_threshold)


def threshold_change(x):
    global threshold_value
    threshold_value = x
    refresh_img(0)


img_raw = cv2.imread('./0.jpg')
# 转灰度
img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
# 二值化
img_threshold = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
# 自适应二值化
# img_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
cv2.imshow('Demo', img_threshold)

# 调节窗口
cv2.namedWindow('Bar')
cv2.createTrackbar('threshold', 'Bar', 0, 255, threshold_change)
cv2.createTrackbar('R', 'Bar', 255, 510, refresh_img)
cv2.createTrackbar('G', 'Bar', 255, 511, refresh_img)
# cv2.createTrackbar('B', 'Bar', 255, 511, refresh_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




