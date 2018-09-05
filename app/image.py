import random

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from skimage import transform
import numpy as np


def read_image(file):
    image = open_image(file)
    image = normalize_image(image)
    return image


def open_image(file):
    image = Image.open(file).convert('L')
    image = image.resize((100, 100))
    return np.array(image)


# Normalization Zero Mean and Unit Variance
def normalize_image(image):
    image = np.asarray(image, np.float32)

    image -= np.mean(image, keepdims=True)
    image /= np.std(image, keepdims=True)

    return img_to_array(image)


def data_augmentation(file, iterations):
    return [transform_image(file) for _ in range(iterations)]


def transform_image(files):
    random_degree = random.uniform(-10, 10)
    file = random.choice(files)

    image = open_image('../data/train/' + file)
    image = transform.rotate(image, random_degree)
    image = normalize_image(image)

    return image
