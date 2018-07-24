import os

from glob import glob
from sklearn import preprocessing, model_selection
from pandas import read_csv
import numpy as np
import cv2

from app import preprocessing as prep
from app.hash import excluded_duplicates


TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'

AUG_RANGE = 20

split_seed = 0

df_train = read_csv('data/train.csv')
df_submission = read_csv('data/submission.csv')

train_files = glob(os.path.join(TRAIN_DIR, '*.jpg'))
test_files = glob(os.path.join(TEST_DIR, '*.jpg'))


def not_fit_images():
    excluded = open('data/not_fit.txt', 'r').read().split('\n')
    return excluded


def single_whales():
    single = []
    frecuency = whale_frecuencies()
    for whale, value in frecuency.items():
        if value <= 1:
            single.append(whale)
    return single


def train_whales():
    excluded = not_fit_images() + excluded_duplicates()
    single = single_whales()
    whales = {}
    dict_t = dict_train()

    for file, whale in dict_t.items():
        if whale != 'new_whale' and whale not in single and file not in excluded:
            whales[file] = whale

    return whales


def whale_frecuencies():
    return df_train['Id'].value_counts().to_dict()


def get_data():
    images, labels = data_augmentation()

    train_images, test_images = split(images)
    train_labels, test_labels = split(labels)

    return train_images, test_images, np.array(train_labels), np.array(test_labels)


def data_augmentation():
    augmented_images, augmented_labels = [], []

    train = train_whales()
    frequency = whale_frecuencies()
    indices, _ = encode_labels(list(train.values()))

    for row, (file, whale) in enumerate(train.items()):
        encoded = indices[row]
        value_frequency = frequency[whale]
        img = get_image(TRAIN_DIR + file)

        if value_frequency < AUG_RANGE:
            augmented_images += prep.data_augmentation(img, AUG_RANGE - value_frequency)
            for _ in range(AUG_RANGE - value_frequency):
                augmented_labels.append(encoded)
        else:
            augmented_images.append(prep.rgb_to_gray(img))
            augmented_labels.append(encoded)

    return np.array(augmented_images), augmented_labels


def images_per_whales():
    train_whale = train_whales()
    dict_t = dict_train()
    associated = {}

    for image, whale in dict_t.items():
        if whale in train_whale:
            if whale not in associated:
                associated[whale] = []
            if image not in associated[whale]:
                associated[whale].append(image)

    return associated


def get_image(file):
    image = cv2.imread(file)
    resize_image = cv2.resize(image, (64, 64))
    return np.array(resize_image)


def encode_labels(labels):
    label_econder = preprocessing.LabelEncoder()
    indices = label_econder.fit_transform(labels)

    return indices, label_econder


def dict_train():
    return dict([(img, whale) for _, img, whale in df_train.to_records()])


def import_eval_images():
    return np.array([get_image(file) for file in test_files])


def save_files():
    train_images, test_images, train_labels, test_labels = get_data()
    evalu = import_eval_images()

    np.save('data/train_images.npy', train_images)
    np.save('data/test_images.npy', test_images)
    np.save('data/eval.npy', evalu)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/test_labels.npy', test_labels)


def import_eval_files():
    evalu = np.load('data/eval.npy')
    dict_eval = dict([(img, whales) for _, img, whales in df_submission.to_records()])
    return evalu, list(dict_eval.keys())


def load_files():
    train_images = np.load('data/train_images.npy')
    test_images = np.load('data/test_images.npy')
    train_labels = np.load('data/train_labels.npy')
    test_labels = np.load('data/test_labels.npy')
    return train_images, test_images, train_labels, test_labels


def split(data):
    return model_selection.train_test_split(data, test_size=0.2, shuffle=False, random_state=split_seed)
