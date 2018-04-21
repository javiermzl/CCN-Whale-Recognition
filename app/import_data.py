import os

from glob import glob
from sklearn import preprocessing
import pandas as pd
import numpy as np
import cv2


TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

df_train = pd.read_csv('data/train.csv')
df_submission = pd.read_csv('data/submission.csv')

train_images = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_images = glob(os.path.join(TEST_DIR, '*.jpg'))

n_classes = 810


def get_image(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(image, (64, 64))
    return np.array(resize_image)


def get_labels():
    df_train['Image'] = df_train['Image'].map(lambda x: '../data/train\\' + x)
    image_label = dict(zip(df_train['Image'], df_train['Id']))
    labels = list(map(image_label.get, train_images))
    indices = preprocessing.LabelEncoder().fit_transform(labels)
    return indices


def import_train_images():
    return np.array([get_image(img) for img in train_images])


def import_eval_images():
    return np.array([get_image(img) for img in test_images])


def save_files():
    train = import_train_images()
    evalu = import_eval_images()

    np.save('../data/train.npy', train)
    np.save('../data/eval.npy', evalu)


def load_files():
    train = np.load('data/train.npy')
    evalu = np.load('data/eval.npy')
    return train, evalu
