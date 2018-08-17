import numpy as np

from app.data import dict_train, encode_labels, TRAIN_DIR, DF_TRAIN
from app.utils.image_hash import excluded_duplicates
from app.image import data_augmentation, read_image


AUG_RANGE = 20


def generate_augmentation():
    augmented_images, augmented_labels = [], []

    train = train_whales()
    frequency = whale_frecuencies()
    indices, _ = encode_labels(list(train.values()))

    for row, (file, whale) in enumerate(train.items()):
        encoded = indices[row]
        value_frequency = frequency[whale]
        img = read_image(TRAIN_DIR + file)

        if value_frequency < AUG_RANGE:
            augmented_images += data_augmentation(img, AUG_RANGE - value_frequency)
            for _ in range(AUG_RANGE - value_frequency):
                augmented_labels.append(encoded)
        else:
            augmented_images.append(img)
            augmented_labels.append(encoded)

    return np.array(augmented_images), augmented_labels


def single_whales():
    single = []
    frecuency = whale_frecuencies()
    for whale, value in frecuency.items():
        if value <= 1:
            single.append(whale)
    return single


def train_whales():
    excluded = excluded_duplicates()
    single = single_whales()
    whales = {}
    dict_t = dict_train()

    for file, whale in dict_t.items():
        if whale != 'new_whale' and whale not in single and file not in excluded:
            whales[file] = whale

    return whales


def whale_frecuencies():
    return DF_TRAIN['Id'].value_counts().to_dict()


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
