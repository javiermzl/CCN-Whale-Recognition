import os

from glob import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing

from app import images


TRAIN_DIR = '../../data/train/'
TEST_DIR = '../../data/test/'

df_train = pd.read_csv('../../data/train.csv')
df_submission = pd.read_csv('../../data/sample_submission.csv')

train_images = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_images = glob(os.path.join(TEST_DIR, '*.jpg'))

df_train['Image'] = df_train['Image'].map(lambda x: TRAIN_DIR+x)
image_label = dict(zip(df_train['Image'], df_train['Id']))

n_classes = 810


def get_labels():
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()

    labels = list(map(image_label.get, train_images))
    labels = le.fit_transform(labels)
    labels = ohe.fit_transform(labels.reshape(-1, 1))

    return labels


def import_train_images():
    return np.array([images.get_image(img) for img in train_images])


def import_test_images():
    return np.array([images.get_image(img) for img in test_images])


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
