import os

from glob import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

df_train = pd.read_csv('../data/train.csv')
df_submission = pd.read_csv('../data/sample_submission.csv')

train_images = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_images = glob(os.path.join(TEST_DIR, '*.jpg'))


def import_image(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(image, (64, 64))
    return np.array(resize_image)


def import_train_data():
    return np.array([import_image(img) for img in train_images])


def import_test_data():
    return np.array([import_image(img) for img in test_images])


if __name__ == '__main__':
    train_np = import_train_data()
    test_np = import_test_data()

    print('\nNP Train Array Size:', train_np.shape)
    print('\nNP Test Array Size:', test_np.shape)

    train_tensor = tf.convert_to_tensor(train_np, np.uint8)
    test_tensor = tf.convert_to_tensor(test_np, np.uint8)

    with tf.Session() as sess:
        print(train_tensor.eval())
        print(test_tensor.eval())
