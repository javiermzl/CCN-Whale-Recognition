import os

from glob import glob
from sklearn import preprocessing, model_selection
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
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(img, (64, 64))
    return np.array(resize_image)


def get_labels():
    labels = dataframe_to_array()
    indices = preprocessing.LabelEncoder().fit_transform(labels)
    return split_train(indices)


def dataframe_to_array():
    df_train['Image'] = df_train['Image'].map(lambda x: '../data/train\\' + x)
    label = dict(zip(df_train['Image'], df_train['Id']))
    return list(map(label.get, train_images))


def import_split_train_images():
    train_img = []
    test_img = []
    train, test = split_train(df_train)

    for file in train_images:
        if train['Image'].str.contains(file[14:]).any():
            img = get_image(file)
            train_img.append(img)

    for file in train_images:
        if test['Image'].str.contains(file[14:]).any():
            img = get_image(file)
            test_img.append(img)

    train = np.array(train_img)
    test = np.array(test_img)

    return train, test


def import_train_images():
    return np.array([get_image(img) for img in train_images])


def import_eval_images():
    return np.array([get_image(img) for img in test_images])


def save_files():
    train, test = import_split_train_images()
    evalu = import_eval_images()

    np.save('data/train.npy', train)
    np.save('data/test.npy', test)
    np.save('data/eval.npy', evalu)


def load_files():
    train = np.load('data/train.npy')
    test = np.load('data/test.npy')
    evalu = np.load('data/eval.npy')
    return train, test, evalu


def split_train(train):
    return model_selection.train_test_split(train, test_size=0.2)