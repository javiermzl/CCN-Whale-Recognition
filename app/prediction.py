import numpy as np
import csv

from app.data import encode_labels, dict_train, test_files

import types


def extract_probabilities(predictions):
    print('Extracting Probabilities')

    if isinstance(predictions, types.GeneratorType):
        predictions = [prob['Probabilities'] for prob in predictions]

    highest = [highest_probabilities(prob) for prob in predictions]

    return highest


def highest_probabilities(prob):
    idx1, prob = minimize_maximum(prob)
    idx2, prob = minimize_maximum(prob)
    idx3, prob = minimize_maximum(prob)
    idx4, prob = minimize_maximum(prob)
    idx5, prob = minimize_maximum(prob)

    return [idx1, idx2, idx3, idx4, idx5]


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
    file = open('data/submission.csv', mode='w', newline='')
    writer = csv.writer(file, delimiter=',')

    print('Writing File')
    writer.writerow(['Image', 'Id'])

    for row, file in enumerate(test_files):
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
