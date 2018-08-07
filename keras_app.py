from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, \
    Flatten, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

import numpy as np
import csv

from app import data


def highest_probabilities(predictions):
    highest = []
    print('Extracting Probabilities')
    for probabilities in predictions:
        idx = np.argmax(probabilities)
        probabilities[idx] = 0
        idx2 = np.argmax(probabilities)
        probabilities[idx2] = 0
        idx3 = np.argmax(probabilities)
        probabilities[idx3] = 0
        idx4 = np.argmax(probabilities)
        probabilities[idx4] = 0
        idx5 = np.argmax(probabilities)

        highest.append([idx, idx2, idx3, idx4, idx5])

    return highest


def reverse_label(probabilities):
    _, encoder = data.encode_labels(list(data.dict_train().values()))
    labels = []
    print('Reversing Labels')
    for p in probabilities:
        labels.append(encoder.inverse_transform(p))
    return labels


def sub_file(pred):
    file = open('data/sub_keras.csv', mode='w', newline='')
    colums = ['Image', 'Id']

    print('Writing File')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(colums)

    for row, i in enumerate(data.test_files):
        ids = str(pred[row][0]) + ' ' + str(pred[row][1]) + ' ' + \
              str(pred[row][2]) + ' ' + str(pred[row][3]) + ' ' + str(pred[row][4])
        result = [i[13:], ids]
        writer.writerow(result)


def get_model():
    model = Sequential()

    model.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0', input_shape=(100, 100, 1)))

    model.add(BatchNormalization(axis=3, name='bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(Flatten())
    model.add(Dense(500, activation="relu", name='rl'))
    model.add(Dropout(0.65))
    model.add(Dense(4251, activation='softmax', name='sm'))

    return model


def train():
    mod = get_model()
    adam = Adam(lr=0.0001)
    mod.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print('Importing Data')
    images, labels = data.gen_raw_data()

    history = mod.fit(images, labels, epochs=150, batch_size=100, verbose=1)
    mod.save('models/keras/whale_model.h5')
    return mod


def recover_model():
    return load_model('models/keras/whale_model.h5')


def predict(mod):
    eval_x, eval_labels = data.import_eval_files()
    predictions = mod.predict(eval_x)
    prob = highest_probabilities(predictions)
    return reverse_label(prob)


mod = train()
# mod = recover_model()
a = predict(mod)
sub_file(a)
