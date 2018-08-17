from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, \
    Flatten, MaxPooling2D, AveragePooling2D, Dropout


NUMBER_CLASSES = 4251


def net():
    network = Sequential()

    network.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0', input_shape=(100, 100, 1)))

    network.add(BatchNormalization(axis=3, name='bn0'))
    network.add(Activation('relu'))

    network.add(MaxPooling2D((2, 2), name='max_pool'))
    network.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    network.add(Activation('relu'))
    network.add(AveragePooling2D((3, 3), name='avg_pool'))

    network.add(Flatten())
    network.add(Dense(500, activation="relu", name='rl'))
    network.add(Dropout(0.65))
    network.add(Dense(NUMBER_CLASSES, activation='softmax', name='sm'))

    return network
