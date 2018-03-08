import os

import tensorflow as tf
import numpy as np

from app import import_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    train_np, test_np = import_data.load()

    print('NP Train Array Size:', train_np.shape)
    print('NP Test Array Size:', test_np.shape)

    train_tensor = tf.convert_to_tensor(train_np, np.uint8)
    test_tensor = tf.convert_to_tensor(test_np, np.uint8)

    with tf.Session() as sess:
        print(train_tensor.eval())
        print(test_tensor.eval())
