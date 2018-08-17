import os

from PIL import Image
from imagehash import phash
from glob import glob


TRAIN_DIR = '../data/train/'
train_files = glob(os.path.join(TRAIN_DIR, '*.jpg'))


def images_size():
    sizes = {}
    for file in train_files:
        sizes[file[14:]] = Image.open(file).size
    return sizes


def image_to_hash():
    hashes = {}
    for file in train_files:
        hashes[file[14:]] = phash(Image.open(file))
    return hashes


def images_per_hash():
    images = {}
    for f, h in image_to_hash().items():
        if h not in images:
            images[h] = []
        if f not in images[h]:
            images[h].append(f)
    return images


def prefered(images, sizes):
    if len(images) == 0:
        return images[0]

    best_image = images[0]
    best_size = sizes[best_image]

    for i in range(1, len(images)):
        image = images[i]
        size = sizes[image]

        if size[0]*size[1] > best_size[0]*best_size[1]:
            best_image = image
            best_size = size

    return best_image


def prefered_images():
    best_images = {}
    sizes = images_size()
    for h, images in images_per_hash().items():
        best_images[h] = prefered(images, sizes)
    return best_images


def excluded_duplicates():
    best_images = list(prefered_images().values())
    excluded = []
    for img in train_files:
        if img[14:] not in best_images:
            excluded.append(img[14:])
    return excluded
