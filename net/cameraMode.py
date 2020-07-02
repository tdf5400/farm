import tensorflow as tf
import cv2 as cv
import roadCal.roadCal as rc
import model
import numpy as np
import time

CPU = True
modelFile = './savedModel/epoch-30'


def cv2tensor(cvImg):
    HEIGHT, WEIGHT, CHANNEL = cvImg.shape

    img = cv.cvtColor(cvImg, cv.COLOR_BGR2RGB)
    img_encode = np.array(cv.imencode('.jpg', img)[1]).tostring()
    img = tf.image.decode_jpeg(img_encode, channels=CHANNEL)

    img_tensor = tf.image.resize_with_pad(image=img, target_height=HEIGHT, target_width=WEIGHT)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255.0

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img = tf.dtypes.cast(img_tensor, dtype=tf.dtypes.float32)
    tensorImg = img / 255.0
    return tensorImg


def test_SingleImg(model, cvImg):
    tensorImg = cv2tensor(cvImg)
    param = model(tensorImg)

    loDiff, upDiff = param[0][0:3], param[0][3:6]
    loDiff = (int(loDiff[0]), int(loDiff[1]), int(loDiff[2]))
    upDiff = (int(upDiff[0]), int(upDiff[1]), int(upDiff[2]))

    try:
        output = rc.cal_floodFill(cvImg, loDiff, upDiff, mask_wide=0)[0]
        return output
    except Exception:
        print('[ERROR] Floodfill failed!')
        return None


if __name__ == '__main__':
    if CPU:
        import os

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    net = model.my_model()
    net.load_weights(filepath=modelFile)
    print('load model success!')

    cap = cv.VideoCapture(0)

    while True:
        # 开始计时
        time_start = time.perf_counter()
        return_val, frame = cap.read()

        frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_CUBIC)

        cv.imshow('cap', frame)

        output = test_SingleImg(net, frame)
        if not output is None:
            cv.imshow('output', output)

        # 显示处理时间
        time_elapsed = (time.perf_counter() - time_start)
        print(f'[console]Used:\t{int(time_elapsed * 1000)} ms', )
        print("[console]Fre:\t%0.2f" % (1 / time_elapsed), " Hz")

        if cv.waitKey(100) & 0xff == ord('q'):
            cv.destroyAllWindows()
            break
