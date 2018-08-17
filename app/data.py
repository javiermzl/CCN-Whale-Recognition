import os
from random import randint

import numpy as np
from glob import glob
from sklearn import preprocessing, model_selection
from pandas import read_csv

from app.image import read_image


TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

train_files = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_files = glob(os.path.join(TEST_DIR, '*.jpg'))

df_train = read_csv('data/train.csv')

# Random Seed pre-generated so both split(images and labels) behave equally
split_seed = randint(0, 99)


def generate_train_files():
    images = np.array([read_image(file) for file in train_files])
    text_labels = np.array(list(dict_train().values()))

    labels, _ = encode_labels(text_labels)

    return images, labels


def generate_train_files_split():
    images, labels = generate_train_files()

    train_images, test_images = split(images)
    train_labels, test_labels = split(labels)

    return train_images, test_images, train_labels, test_labels


def generate_eval_images():
    return np.array([read_image(file) for file in test_files])


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
    return dict([(img, whale) for _, img, whale in df_train.to_records()])


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
    return model_selection.train_test_split(data, test_size=0.1, shuffle=False, random_state=split_seed)
