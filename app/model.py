import tensorflow as tf

from app import data


def conv_net(features, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

    # Convolutional Layers
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )
    conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense Layers
    dense = tf.layers.flatten(conv2)

    dense = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.relu)
    dense = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    output = tf.layers.dense(inputs=dense, units=data.n_classes)

    return output


def model_fn(features, labels, mode):
    logits = conv_net(features, mode)

    loss, eval_metrics_ops, train_op = None, None, None
    predictions = {
        'Classes': tf.argmax(logits, axis=1),
        'Probabilities': tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=data.n_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics_ops = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['Classes'])
            }

    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = predictions

    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics_ops, predictions=predictions
    )

    return estimator_spec
