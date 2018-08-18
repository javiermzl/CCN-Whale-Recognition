import os

import tensorflow as tf

from app.data import generate_train_files, generate_eval_images
from app.model import model_fn
from app.prediction import generate_submission


TRAIN_MODEL = True


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Remove Warnings
tf.logging.set_verbosity(tf.logging.INFO)   # Show Progress Info


def model():
    return tf.estimator.Estimator(model_fn, model_dir='models/tensor/')


def train_model(x, y):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x,
        y=y,
        batch_size=100,
        shuffle=True,
        num_epochs=100,
    )
    model.train(input_fn=input_fn)


def evaluate_model(x, y):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x,
        y=y,
        batch_size=100,
        num_epochs=1,
        shuffle=False,
    )
    e = model.evaluate(input_fn=input_fn)
    print("Testing Accuracy:", e['accuracy'])


def predict(images):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=images,
        num_epochs=1,
        shuffle=False
    )

    return model.predict(input_fn=input_fn)


if __name__ == '__main__':
    model = model()

    if TRAIN_MODEL:
        # Split for Train and Eval -> generate_train_files_split
        train_images, train_labels = generate_train_files()

        train_model(train_images, train_labels)
        evaluate_model(train_images, train_labels)

    eval_images = generate_eval_images()
    predictions = predict(eval_images)
    generate_submission(predictions)
