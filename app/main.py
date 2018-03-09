import os

import tensorflow as tf

from app import import_data as data
from app import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == '__main__':
    train_np, test_np = data.load()
    y = data.get_labels()

    model = tf.estimator.Estimator(model.model_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': train_np},
        y=y,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    model.train(input_fn, steps=1000)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': test_np},
        y=y,
        batch_size=100,
        shuffle=False)

    e = model.evaluate(input_fn)
    print("Testing Accuracy:", e['accuracy'])
