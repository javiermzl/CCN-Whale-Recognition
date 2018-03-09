import numpy as np
import cv2


def get_image(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(image, (64, 64))
    return np.array(resize_image)
