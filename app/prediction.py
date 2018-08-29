import types

import numpy as np
import csv

from app.data import encode_labels, dict_train, TEST_FILES


def extract_probabilities(predictions):
    print('Extracting Probabilities')

    # TensorFlow Prediction Type
    if isinstance(predictions, types.GeneratorType):
        predictions = [prob['Probabilities'] for prob in predictions]

    highest = [highest_probabilities(prob) for prob in predictions]

    return highest


def highest_probabilities(prob):
    probabilities = []

    for _ in range(5):
        maximum, prob = minimize_maximum(prob)
        probabilities.append(maximum)

    return probabilities


def minimize_maximum(array):
    maximum = np.argmax(array)
    array[maximum] = 0
    return maximum, array


def reverse_labels(probabilities):
    print('Reversing Labels')

    _, encoder = encode_labels(list(dict_train().values()))
    labels = [encoder.inverse_transform(prob) for prob in probabilities]

    return labels


def create_file(predictions):
    file = open('data/output.csv', mode='w', newline='')
    writer = csv.writer(file, delimiter=',')

    print('Writing File')
    writer.writerow(['Image', 'Id'])

    for row, file in enumerate(TEST_FILES):
        ids = format_row(predictions[row])
        file_name = file[13:]

        writer.writerow([file_name, ids])


def format_row(values):
    return str(values[0]) + ' ' + str(values[1]) + ' ' + \
        str(values[2]) + ' ' + str(values[3]) + ' ' + str(values[4])


def generate_submission(predictions):
    probabilities = extract_probabilities(predictions)
    labels = reverse_labels(probabilities)
    create_file(labels)
