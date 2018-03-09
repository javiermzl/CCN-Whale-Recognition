import tensorflow as tf

from app import import_data as data


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
    logits_train = conv_net(features, True)
    logits_test = conv_net(features, False)

    predict_classes = tf.arg_max(logits_test, axis=1)
    predict_prob = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predict_classes)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=data.n_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits_train)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict_classes)

    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=accuracy
    )

    return estimator_spec
