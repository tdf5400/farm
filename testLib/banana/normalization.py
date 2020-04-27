# 图片尺寸归一化

import cv2 as cv
img_mountStart = 0  # 照片起始编号
img_mountStop = 11  # 照片结束编号
img_fileLocate = ".\\"  # 路径
img_form = ".jpg"   # 照片格式
img_TargetSize = (640, 480) # 尺寸

if __name__=="__main__":
    print("开始转换！")

    # img = cv.imread(img_fileLocate + str(0) + img_form)
    # cv.imshow("demo", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    for i in range(img_mountStart, (img_mountStop+1)):
        img = cv.imread(img_fileLocate + str(i) + img_form)
        try:
            if img is None:
                print("[Fail]Empty Image!", end=' ')
                print(img_fileLocate + str(i) + img_form)
                continue
            # if img.shape[2] == 1:
            #     print("[Fail]Already Operate!", end=' ')
            #     print(img_fileLocate + str(i) + img_form)
            # else:
            #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            if not img_TargetSize == (0,0):
                img = cv.resize(img, img_TargetSize)

            cv.imwrite(img_fileLocate+str(i)+img_form, img)

            print("[Succeed]", end=' ')
            print(img_fileLocate + str(i) + img_form)
        except Exception:
            print("[Fail]", end=' ')
            print(img_fileLocate+str(i)+img_form)