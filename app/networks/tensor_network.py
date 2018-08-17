import tensorflow as tf


def net(inputs, mode):
    logits = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[7, 7],
        padding="same",
    )

    logits = tf.layers.batch_normalization(logits, axis=3)
    logits = tf.nn.relu(logits)

    logits = tf.layers.max_pooling2d(inputs=logits, pool_size=[2, 2], strides=[2, 2])
    logits = tf.layers.conv2d(
        inputs=logits,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    logits = tf.layers.average_pooling2d(inputs=logits, pool_size=[3, 3], strides=3)

    logits = tf.layers.flatten(logits)
    logits = tf.layers.dense(inputs=logits, units=500, activation=tf.nn.relu)
    logits = tf.layers.dropout(
        inputs=logits, rate=0.65, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    logits = tf.layers.dense(
        inputs=logits, units=4251, activation=tf.nn.softmax
    )

    return logits
