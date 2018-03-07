import os

from glob import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

train_csv = pd.read_csv('../data/train.csv')
submission = pd.read_csv('../data/sample_submission.csv')

train_images = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_images = glob(os.path.join(TEST_DIR, '*.jpg'))


def check_data():
    print('Train Size:', len(train_images))
    print('Test Size:', len(test_images))

    print(train_csv.shape)
    print(submission.shape)


def import_image(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (64, 64))
    return np.array(resized_image)


def import_train_data():
    return np.array([import_image(img) for img in train_images])


if __name__ == '__main__':
    check_data()

    train = import_train_data()
    print('\nNP Array Size:', train.shape)

    train_tf = tf.convert_to_tensor(train, np.uint8)
