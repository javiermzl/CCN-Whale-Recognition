from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_images(imgs, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i], cmap='gray')
        plt.show()


def zoom_images(img):
    img = image.random_zoom(
        img, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
    ) * 255
    return np.array(img)


def rotate_images(img):
    img = image.random_rotation(
        img, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
    ) * 255
    return np.array(img)


def transform_image(img):
    img = rotate_images(img)
    img = zoom_images(img)
    img = rgb_to_gray(img)
    return img


def rgb_to_gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def data_augmentation():
    read = cv2.imread('../data/train/00aa021c.jpg')
    resize_image = cv2.resize(read, (64, 64))
    imgs = np.array(resize_image)

    imgs = [transform_image(imgs) * 255 for _ in range(5)]

    plot_images(imgs, None, rows=1)
