import os

from glob import glob
import pandas as pd
import numpy as np
import cv2

TRAIN_DIR = '../../data/train/'
TEST_DIR = '../../data/test/'

df_train = pd.read_csv('../../data/train.csv')
df_submission = pd.read_csv('../../data/sample_submission.csv')

train_images = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_images = glob(os.path.join(TEST_DIR, '*.jpg'))


def get_image(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(image, (64, 64))
    return np.array(resize_image)


def import_train_images():
    return np.array([get_image(img) for img in train_images])


def import_test_images():
    return np.array([get_image(img) for img in test_images])


def save(train, test):
    np.save('../data/train.npy', train)
    np.save('../data/test.npy', test)


def load():
    train = np.load('../data/train.npy')
    test = np.load('../data/test.npy')
    return train, test


if __name__ == '__main__':
    train_np = import_train_images()
    test_np = import_test_images()

    save(train_np, test_np)
