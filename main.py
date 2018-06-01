import os

import tensorflow as tf
import numpy as np

from app import data
from app.model import model_fn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Remove Warnings
tf.logging.set_verbosity(tf.logging.INFO)   # Show Progress Info

train_images, test_images, train_labels, test_labels = data.load_files()

train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)


input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_images,
    y=train_labels,
    batch_size=128,
    shuffle=True,
    num_epochs=None
)

model = tf.estimator.Estimator(model_fn, model_dir='model/')
model.train(input_fn, steps=10000)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_images,
    y=test_labels,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
)

e = model.evaluate(input_fn)
print("Testing Accuracy:", e['accuracy'])
