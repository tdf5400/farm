import numpy as np
import cv2


"""
Main
"""
cap = cv2.VideoCapture(1)

# 读取快门数
log = np.load('./testLib/camera/log.npz')
number = log['order_number']

while True:
    ret, frame = cap.read()

    # 显示图像
    cv2.imshow('Cap', frame)

    # 按键动作
    key = cv2.waitKey(1)
    if key == 27:           # Esc - 退出
        break
    elif key == ord('s'):   # s   - 保存图像
        cv2.imwrite('./testLib/camera/'+str(number)+'.jpg', frame)
        print(f"Save as: {str(number)+'.jpg'}!")
        number += 1

# 退出程序
number = 0  # 清空读数
np.savez('./testLib/camera/log.npz', order_number=number)
cap.release()
cv2.destroyAllWindows()
