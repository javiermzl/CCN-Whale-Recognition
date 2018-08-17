import os

import tensorflow as tf

from app.data import generate_train_files, generate_eval_images
from app.model import model_fn
from app.prediction import generate_submission


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Remove Warnings
tf.logging.set_verbosity(tf.logging.INFO)   # Show Progress Info


if __name__ == '__main__':
    train, train_labels = generate_train_files()

    model = tf.estimator.Estimator(model_fn, model_dir='models/tensor/')

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

    eval_images = generate_eval_images()

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_images,
        num_epochs=1,
        shuffle=False
    )
    predictions = model.predict(input_fn=input_fn)
    generate_submission(predictions)
