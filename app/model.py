import tensorflow as tf

N_CLASSES = 810
NUM_CHANNELS = 32


def conv_net(features, mode):
    # Input Layer
    output = tf.reshape(features, [-1, 64, 64, 1])

    channels = [NUM_CHANNELS, NUM_CHANNELS * 2]
    for i, c in enumerate(channels):
        # Convolutional Layers
        output = tf.layers.conv2d(
            inputs=output,
            filters=c,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu,
        )
        output = tf.layers.max_pooling2d(inputs=output, pool_size=2, strides=2)

    # Dense Layers
    dense = tf.layers.flatten(output)

    dense = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    output = tf.layers.dense(inputs=dense, units=N_CLASSES)
    output = tf.nn.l2_normalize(output, 0)

    return output


def model_fn(features, labels, mode):
    embeddings = conv_net(features, mode)

    loss, eval_metrics_ops, train_op = None, None, None
    predictions = {
        'Classes': tf.argmax(embeddings, axis=1),
        'Probabilities': tf.nn.softmax(embeddings)
    }

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
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
        mode=mode, loss=loss, train_op=train_op,
        eval_metric_ops=eval_metrics_ops, predictions=predictions
    )

    return estimator_spec
