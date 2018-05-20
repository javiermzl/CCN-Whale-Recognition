import os

from random import randint
from glob import glob
from sklearn import preprocessing, model_selection
import pandas as pd
import numpy as np
import cv2

from app import preprocessing as prep


TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

#split_seed = randint(0, 100)
split_seed = 0  # During Dev

df_train = pd.read_csv('data/train.csv')
df_submission = pd.read_csv('data/submission.csv')

train_images = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_images = glob(os.path.join(TEST_DIR, '*.jpg'))


def get_data():
    images, labels = data_augmentation()

    train_img, test_img = split(images)
    train_labels, test_labels = split(labels)

    return train_img, test_img, np.array(train_labels), np.array(test_labels)


def data_augmentation():
    augmented_images, augmented_labels = [], []
    row = 0

    labels = get_labels()
    frequency = df_train['Id'].value_counts().to_dict()

    for file in train_images:

        value_frequency = frequency[df_train.iloc[row]['Id']]
        label = labels[row]
        img = get_image(file)

        if value_frequency < 10:
            augmented_images += prep.data_augmentation(img, 10 - value_frequency)
            for i in range(10 - value_frequency):
                augmented_labels.append(label)
        else:
            augmented_images.append(prep.rgb_to_gray(img))
            augmented_labels.append(label)

        row += 1

    return np.array(augmented_images), augmented_labels


def get_image(file):
    img = cv2.imread(file)
    resize_image = cv2.resize(img, (64, 64))
    return np.array(resize_image)


def get_labels():
    labels = dataframe_to_array()
    indices = preprocessing.LabelEncoder().fit_transform(labels)
    return indices


def dataframe_to_array():
    df_train['Image'] = df_train['Image'].map(lambda x: '../data/train\\' + x)
    label = dict(zip(df_train['Image'], df_train['Id']))
    return list(map(label.get, train_images))


def import_eval_images():
    return np.array([get_image(img) for img in test_images])


def save_files():
    train_np, test_np, train_labels, test_labels = get_data()
    evalu = import_eval_images()

    np.save('data/train_images.npy', train_np)
    np.save('data/test_images.npy', test_np)
    np.save('data/eval.npy', evalu)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/test_labels.npy', test_labels)


def load_files():
    train = np.load('data/train_images.npy')
    test = np.load('data/test_images.npy')
    evalu = np.load('data/eval.npy')
    train_labels = np.load('data/train_labels.npy')
    test_labels = np.load('data/test_labels.npy')
    return train, test, evalu, train_labels, test_labels


def split(data):
    return model_selection.train_test_split(data, test_size=0.2, shuffle=False, random_state=split_seed)
