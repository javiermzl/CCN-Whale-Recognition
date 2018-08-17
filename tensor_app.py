import os

import tensorflow as tf

from app.model import model_fn
from app.data import import_eval_files, gen_raw_data
from app.prediction import generate_submission


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Remove Warnings
tf.logging.set_verbosity(tf.logging.INFO)   # Show Progress Info


if __name__ == '__main__':
    train, train_labels = gen_raw_data()

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

    eval_images, eval_labels = import_eval_files()

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_images,
        num_epochs=1,
        shuffle=False
    )
    predictions = model.predict(input_fn=input_fn)
    generate_submission(predictions)
