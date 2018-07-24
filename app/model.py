import tensorflow as tf

from app import resnet

N_CLASSES = 1976    # - >len(set(train_labels))


class WhaleModel(resnet.Model):
    def __init__(self, resnet_size, data_format=None,
                 dtype=resnet.DEFAULT_DTYPE):

        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(WhaleModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            num_classes=N_CLASSES,
            num_filters=16,
            kernel_size=5,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            data_format=data_format,
            dtype=dtype
        )


def learning_rate_with_decay(batch_size, batch_denom, num_images, boundary_epochs, decay_rates):

    initial_learning_rate = 0.000001 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def model_fn(features, labels, mode):
    features = tf.reshape(features, [-1, 64, 64, 1])
    model = WhaleModel(resnet_size=32)
    logits = model(features, mode)

    learning_rate_fn = learning_rate_with_decay(
        batch_size=128, batch_denom=128,
        num_images=50000, boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    loss, eval_metrics_ops, train_op = None, None, None
    predictions = {
        'Classes': tf.argmax(logits, axis=1),
        'Probabilities': tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            learning_rate = learning_rate_fn(global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=global_step
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
