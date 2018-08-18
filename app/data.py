import os
from random import randint

import numpy as np
from glob import glob
from sklearn import preprocessing, model_selection
from pandas import read_csv

from app.image import read_image


# Random Seed pre-generated so both split(images and labels) behave equally
SPLIT_SEED = randint(0, 99)

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

TRAIN_FILES = glob(os.path.join(TRAIN_DIR, '*.jpg'))
TEST_FILES = glob(os.path.join(TEST_DIR, '*.jpg'))

DF_TRAIN = read_csv('data/train.csv')


def generate_train_files():
    print('Importing Data')

    images = np.array([read_image(file) for file in TRAIN_FILES])
    text_labels = np.array(list(dict_train().values()))

    labels, _ = encode_labels(text_labels)

    return images, labels


def generate_train_files_split():
    images, labels = generate_train_files()

    train_images, test_images = split(images)
    train_labels, test_labels = split(labels)

    return train_images, test_images, train_labels, test_labels


def generate_eval_images():
    return np.array([read_image(file) for file in TEST_FILES])


def encode_labels(labels):
    encoder = preprocessing.LabelEncoder()

    int_labels = encoder.fit_transform(labels)
    one_hot_labels = one_hot_encode(int_labels)

    return one_hot_labels, encoder


def one_hot_encode(labels):
    encoder = preprocessing.OneHotEncoder(sparse=False)

    reshaped_labels = labels.reshape(len(labels), 1)
    one_hot_labels = encoder.fit_transform(reshaped_labels)

    return one_hot_labels


def dict_train():
    return dict([(img, whale) for _, img, whale in DF_TRAIN.to_records()])


def save_files():
    train_images, train_labels = generate_train_files()
    eval_images = generate_eval_images()

    np.save('data/images/train_images.npy', train_images)
    np.save('data/images/train_labels.npy', train_labels)
    np.save('data/images/eval_images.npy', eval_images)


def load_train_files():
    train_images = np.load('data/images/train_images.npy')
    train_labels = np.load('data/images/train_labels.npy')
    return train_images, train_labels


def load_eval_images():
    return np.load('data/images/train_images.npy')


def split(data):
    return model_selection.train_test_split(data, test_size=0.1, shuffle=False, random_state=SPLIT_SEED)
