from tensorflow.python.keras.preprocessing.image import random_rotation, random_zoom
import numpy as np


def zoom_image(img):
    img = random_zoom(
        img, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
    ) * 255
    return np.array(img)


def rotate_image(img):
    img = random_rotation(
        img, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
    ) * 255
    return np.array(img)


def transform_image(img):
    img = rotate_image(img)
    img = zoom_image(img)
    # img = rgb_to_gray(img)
    return img


def rgb_to_gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def data_augmentation(img, iterations):
    return [transform_image(img) * 255 for _ in range(iterations)]