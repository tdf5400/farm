import just_one_way
import roadCal.roadCal as rc
import time
import cv2

imgPath = "../../testLib/banana/"
output_path = "./output/"  # 输出计算结果的路径
imgBegin = 1  # 图片开始的编号
imgEnd = 31  # 图片结束的编号

CAL_WAY = 1  # 计算方法 0-kmeans, 1-HSV, 大于2-Floodfill
OUTPUT = 2  # 输出结果 0-DISABLE 1-ENABLE 2-输出与原图像的与计算结果

if __name__ == "__main__":
    time_start = time.perf_counter()    # 开始计时
    count = 0                           # 图片计数

    for i in range(imgBegin, imgEnd+1):
        img = cv2.imread(imgPath + str(i) + ".jpg", cv2.IMREAD_COLOR)
        if img is None:  # 跳过无图片的路径
            print("[Error]No Image!")
            continue
        count += 1       # 图片计数+1
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if CAL_WAY == 0:  # k-means
            output = just_one_way.cal_kmeans(img)
        elif CAL_WAY == 1:  # HSV
            output = just_one_way.HSV_threshold(img, (300, 440, 340, 480), (20, 150, 155))
        else:  # floodfill
            output = rc.cal_floodFill(img, (18, 30, 100), (18, 30, 50), mask_wide=230)[1]
        assert (output is not None), f"[Error]{i}.jpg Calculate Fail,Please check your param!"


        if OUTPUT:  # 保存图片
            if OUTPUT == 2:  # 与原图像相与结果
                for j in range(output.shape[0]):
                    for k in range(output.shape[1]):
                        if output.item(j, k) == 255:  # output白色区域在img画成红色
                            img[j, k] = (0, 0, 255)
                output = img
            cv2.imwrite(output_path + str(i) + "-" + str(CAL_WAY) + ".jpg", output)
        # cv2.imshow('output', output)
        cv2.waitKey(0)

    # 显示处理时间
    time_elapsed = (time.perf_counter() - time_start)
    print(f'[console]Used:\t{int(time_elapsed * 1000)} ms', )
    print(f'[console]Count:\t{count}')
    print("[console]Fre:\t%0.2f" % (count / time_elapsed), " Hz")
