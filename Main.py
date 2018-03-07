import os

from glob import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def import_train_data():
    images = []
    for _file in train_images:
        img = cv2.imread(_file)
        images.append(img)
    return images


if __name__ == '__main__':
    check_data()

    train = import_train_data()
    print('\nNP Array Size:', len(train))
