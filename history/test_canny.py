import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
start = time.perf_counter()

img = cv2.resize(cv2.imread('./testLib/testRoad_1.jpg', cv2.IMREAD_GRAYSCALE), (640, 480))
cv2.imshow('Raw', img)

result = cv2.Canny(img, 140, 180)
# result = cv2.HoughLines(img, )

cv2.imshow('Result', result)

elapsed = (time.perf_counter() - start)
print("Time used:", elapsed)
cv2.waitKey(0)
cv2.destroyAllWindows()

