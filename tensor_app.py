import os

import tensorflow as tf
import numpy as np
import csv

from app.model import model_fn
from app import data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Remove Warnings
tf.logging.set_verbosity(tf.logging.INFO)   # Show Progress Info


def highest_probabilities(predictions):
    highest = []
    print('Extracting Probabilities')
    for p in predictions:
        probabilities = p['Probabilities']
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
    file = open('data/sub_tensor.csv', mode='w', newline='')
    colums = ['Image', 'Id']

    print('Writing File')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(colums)

    for row, i in enumerate(data.test_files):
        ids = str(pred[row][0]) + ' ' + str(pred[row][1]) + ' ' + \
              str(pred[row][2]) + ' ' + str(pred[row][3]) + ' ' + str(pred[row][4])
        result = [i[13:], ids]
        writer.writerow(result)


if __name__ == '__main__':
    train, train_labels = data.gen_raw_data()

    model = tf.estimator.Estimator(model_fn, model_dir='models/whale_model/')

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train,
        y=train_labels,
        batch_size=100,
        shuffle=True,
        num_epochs=100,
    )

    model.train(input_fn=input_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train,
        y=train_labels,
        batch_size=100,
        num_epochs=1,
        shuffle=False,
    )
    e = model.evaluate(input_fn=input_fn)
    print("Testing Accuracy:", e['accuracy'])

    eval_images, eval_labels = data.import_eval_files()
    # eval_images = eval_images.astype(np.float32)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_images,
        num_epochs=1,
        shuffle=False
    )
    predictions = model.predict(input_fn=input_fn)
    h_p = highest_probabilities(predictions)
    h_l = reverse_label(h_p)
    sub_file(h_l)
