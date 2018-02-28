import os

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


def check_data():
    print('Train Size:', len(os.listdir(TRAIN_DIR)))
    print('Test Size:', len(os.listdir(TEST_DIR)))

    print(train_csv.shape)
    print(submission.shape)


def import_data():
    images = []
    for _file in os.listdir(TRAIN_DIR):
        img = cv2.imread(os.path.join(TRAIN_DIR, _file))
        images.append(img)
    return images


if __name__ == '__main__':
    check_data()
    train = import_data()
    print('\nTamaño NP Array:', len(train))
