from app import hash
from app import data


def whales_associated():
    dict_train = data.dict_train()
    img_per_hash = hash.image_to_hash()
    associated = {}

    for image, whale in dict_train.items():
        if whale != 'new_whale':
            h = img_per_hash[image]
            if h not in associated:
                associated[h] = []
            if whale not in associated[h]:
                associated[h].append(whale)
    return associated


def group_hash():
    hashes = hash.image_to_hash()
    img_per_hashes = hash.images_per_hash()

    for img, h in hashes.items():
        h = str(h)
        if h in img_per_hashes:
            h = img_per_hashes[h]
        hashes[img] = h

    return hashes
