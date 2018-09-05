import os
from random import randint

import numpy as np
from glob import glob
from sklearn import preprocessing, model_selection
from pandas import read_csv

from app.image import read_image, data_augmentation


SPLIT_SEED = randint(0, 99)
MAX_AUG_RANGE = 30

TRAIN_FILES = glob(os.path.join('../data/train/', '*.jpg'))
TEST_FILES = glob(os.path.join('../data/test/', '*.jpg'))

DF_TRAIN = read_csv('data/train.csv')


def whale_frecuencies():
    return DF_TRAIN['Id'].value_counts().to_dict()


def images_per_whales(whales_dict):
    whales = {}
    for image, whale in whales_dict.items():
        if whale not in whales:
            whales[whale] = []
        if image not in whales[whale]:
            whales[whale].append(image)
    return whales


def augmented_data():
    images, labels = [], []
    frequency = whale_frecuencies()
    train_dict = train_dictionary()
    indices, _ = encode_labels(np.array(list(train_dict.values())))
    img_per_whale = images_per_whales(train_dict)

    for row, (file, whale) in enumerate(train_dict.items()):
        label = indices[row]
        image = read_image('../data/train/' + file)

        images.append(image)
        labels.append(label)

        if frequency[whale] < MAX_AUG_RANGE:
            aug_range = MAX_AUG_RANGE - frequency[whale] - 1

            images += data_augmentation(img_per_whale[whale], aug_range)
            for _ in range(aug_range):
                labels.append(label)

            frequency[whale] = MAX_AUG_RANGE

    return np.array(images), np.array(labels)


def train_files():
    print('Importing Train Data')

    images = np.array([read_image(file) for file in TRAIN_FILES])
    text_labels = np.array(list(train_dictionary().values()))

    labels, _ = encode_labels(text_labels)

    return images, labels


def train_eval_files():
    images, labels = load_train_files()

    train_images, eval_images = split(images)
    train_labels, eval_labels = split(labels)

    return train_images, train_labels, eval_images, eval_labels


def test_images():
    print('Importing Test Data')
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


def train_dictionary():
    return dict([(img, whale) for _, img, whale in DF_TRAIN.to_records()])


def save_files():
    train_images, train_labels = train_files()
    eval_images = test_images()

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
