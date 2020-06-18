import tensorflow as tf

def my_model():

    model = tf.keras.models.Sequential([
        # 1
        tf.keras.layers.Conv2D(filters=32, kernel_size=(11, 11), strides=(1, 1), activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        # 2
        tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(3, 3), activation='relu', padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        # 3
        #tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'),
        #tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        #tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # 4
        #tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'),
        # 5
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(units=2048, activation='relu'),
        #tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Dense(units=4096, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        # 6 output
        tf.keras.layers.Dense(units=6)
    ])
#     fc_2 = tf.keras.layers.BatchNormalization()
# #     fc_3 = tf.keras.layers.Dense(units='classes', activation='softmax')(fc_2)
# #     fc_3.add(tf.keras.layers.BatchNormalization())
#     data_output = (fc_2)
#     data_output = tf.keras.layers.BatchNormalization()(data_output)

    return model


def model_loss(label, pre):
    assert pre.shape == label.shape, 'Incompatibel shapes! Please input same shape data!'
    data_length = pre.shape[0]
    channel = pre.shape[1]
    loss = []

    maes = tf.losses.mean_absolute_error(label, pre)
    loss = tf.reduce_sum(maes)

    # for i in range(data_length):
    #     single_loss = 0
    #     for j in range(channel):
    #         single_loss += tf.math.square(label[i][j] - pre[i][j])
    #     loss.append(single_loss)
    #
    # tf.math.reduce_mean(loss)
    # print('loss:{}'.format(loss))
    return loss

if __name__ == '__main__':


    model = my_model()
    print('Run successfully!')
