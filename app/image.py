from tensorflow.python.keras.preprocessing.image import random_rotation, random_zoom
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np


def read_image(file):
    image = open_image(file)
    image = normalize_image(image)
    return image


def open_image(file):
    image = Image.open(file).convert('L')
    image = image.resize((100, 100))
    return img_to_array(image)


# Normalization Zero Mean and Unit Variance
def normalize_image(image):
    image -= np.mean(image, keepdims=True)
    image /= np.std(image, keepdims=True)
    return image


def data_augmentation(img, iterations):
    return [transform_image(img) for _ in range(iterations)]


def transform_image(img):
    img = rotate_image(img)
    img = zoom_image(img)
    return img


def rotate_image(img):
    img = random_rotation(
        img, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
    ) * 255
    return np.array(img)


def zoom_image(img):
    img = random_zoom(
        img, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
    ) * 255
    return np.array(img)

