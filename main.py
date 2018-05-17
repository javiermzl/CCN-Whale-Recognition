import os

import tensorflow as tf
import numpy as np

from app import data, model, preprocessing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Remove Warnings
tf.logging.set_verbosity(tf.logging.INFO)   # Show Progress Info


train_np, test_np, eval_np = data.load_files()
y_train, y_test = data.get_labels()

train_np = train_np.astype(np.float32)
test_np = test_np.astype(np.float32)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_np,
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True,
)

model = tf.estimator.Estimator(model.model_fn)
model.train(input_fn, steps=10000)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_np,
    y=y_test,
    batch_size=100,
    num_epochs=1,
    shuffle=False
)

e = model.evaluate(input_fn=input_fn)
print("Testing Accuracy:", e['accuracy'])
