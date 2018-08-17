import os

from glob import glob
from sklearn import preprocessing, model_selection
from pandas import read_csv
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


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

encoder = None


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
    dict_t = dict_train()
    associated = {}
    t_w = train_whales().values()

    for image, whale in dict_t.items():
        if whale in t_w:
            if whale not in associated:
                associated[whale] = []
            if image not in associated[whale]:
                associated[whale].append(image)

    return associated


def triplet_gen():
    img_per_whales = images_per_whales()
    positive, anchor, negative = [], [], []

    for whale, images in img_per_whales.items():
        p, n = whale, images
        for img in images:
            while p is whale:
                p = np.random.choice(images)
            while n is images:
                n = np.random.choice(list(img_per_whales.values()))

            anchor.append(img)
            positive.append(p)
            negative.append(np.random.choice(n))

    return anchor, positive, negative


def triplet_labels():
    img_per_whales = images_per_whales()
    labels = []
    for whale, images in img_per_whales.items():
        for _ in images:
            labels.append(whale)
    return np.array(labels)


def triplet_images():
    anchor, positive, negative = triplet_gen()

    anchor = read_triplets(anchor)
    positive = read_triplets(positive)
    #negative = read_triplets(negative)

    return anchor, positive


def read_triplets(images):
    img = []
    for image in images:
        img.append(get_image(TRAIN_DIR + image, False))
    return img


def gen_data():
    a, p = triplet_images()
    a, p = np.array(a), np.array(p)

    labels = triplet_labels()

    labels, _ = encode_labels(labels)

    train_dict = {'Anchor': a, 'Positive': p}

    return train_dict, labels


def gen_raw_data():
    a = []
    for img in train_files:
        a.append(get_image(img, False))
    a = np.array(a)
    labels = np.array(list(dict_train().values()))

    labels, _ = encode_labels(labels)

    return a, labels


def get_image(file, process):
    image = Image.open(file).convert('L')
    image = image.resize((100, 100))
    image = img_to_array(image)

    if process:
        image = prep.transform_image(image)

    # Normalization
    image -= np.mean(image, keepdims=True)
    image /= np.std(image, keepdims=True)

    return image


def encode_labels(labels):
    label_econder = preprocessing.LabelEncoder()
    indices = label_econder.fit_transform(labels)

    ohe = preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = indices.reshape(len(indices), 1)
    onehot_encoded = ohe.fit_transform(integer_encoded)

    return onehot_encoded, label_econder


def dict_train():
    return dict([(img, whale) for _, img, whale in df_train.to_records()])


def import_eval_images():
    return np.array([get_image(file, False) for file in test_files])


def save_files():
    train_images, test_images, train_labels, test_labels = get_data()
    evalu = import_eval_images()

    np.save('data/train_images.npy', train_images)
    np.save('data/test_images.npy', test_images)
    np.save('data/eval.npy', evalu)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/test_labels.npy', test_labels)


def import_eval_files():
    evalu = import_eval_images()
    dict_eval = dict([(img, whales) for _, img, whales in df_submission.to_records()])
    return evalu, list(dict_eval.keys())


def load_files():
    train_images = np.load('data/train_images.npy')
    test_images = np.load('data/test_images.npy')
    train_labels = np.load('data/train_labels.npy')
    test_labels = np.load('data/test_labels.npy')
    return train_images, test_images, train_labels, test_labels


def split(data):
    return model_selection.train_test_split(data, test_size=0.1, shuffle=False, random_state=split_seed)
