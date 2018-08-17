import tensorflow as tf
from tensorflow.keras.backend import categorical_crossentropy
from app.networks.tensor_network import net


IMAGE_SIZE = 100


def model_fn(features, labels, mode):
    features = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    logits = net(features, mode)

    loss, eval_metrics_ops, train_op, logging_hook = None, None, None, None
    predictions = {
        'Classes': tf.argmax(logits, axis=1),
        'Probabilities': tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(categorical_crossentropy(
            tf.cast(labels, logits.dtype), logits
        ))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=1e-7)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions['Classes'])
        eval_metrics_ops = {'accuracy': accuracy}

    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = predictions

    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op,
        eval_metric_ops=eval_metrics_ops, predictions=predictions
    )
    return estimator_spec
