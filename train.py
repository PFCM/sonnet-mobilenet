import tensorflow as tf
import sonnet as snt

import tensorflow.contrib.tfprof as tfprof
from tensorflow.python import debug as tfdbg

from mobilenet import MobileNet
from dogs_dataset import dog_tensor


tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size for sgd')
tf.app.flags.DEFINE_float('alpha', 1.0, 'width scale for MobileNet')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'step size for sgd')

tf.app.flags.DEFINE_string('logdir', '/tmp/mobilenet', 'where to save things')
tf.app.flags.DEFINE_string('dogdir', None, 'where the dogs are')
tf.app.flags.DEFINE_string('dog_regex', 'retriever', 'regex to filter classes')
tf.app.flags.DEFINE_integer('max_steps', 10000, 'how many batches to train on')

FLAGS = tf.app.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.variable_scope('data'):
        train_images, train_labels, num_classes = dog_tensor(
            FLAGS.dogdir, FLAGS.batch_size, class_regex=FLAGS.dog_regex)

    net = MobileNet(num_classes, alpha=FLAGS.alpha)

    train_logits = net(train_images, is_training=True)
    tf.logging.info('built model on training data')
    param_stats = tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    )

    with tf.variable_scope('training'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels,
                                                      logits=train_logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('train/xent', loss)

        global_step = tf.train.get_or_create_global_step()

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_step = opt.minimize(loss, global_step=global_step)
        # make sure we update running averages
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            train_step = tf.group(train_step, *update_ops)

    # valid/test

    sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=global_step,
                             save_summaries_secs=15)

    with sv.managed_session() as sess, sv.stop_on_exception():
        tf.logging.debug('ready to run things')

        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)

        step = sess.run(global_step)
        while step < FLAGS.max_steps:
            step, train_loss, _ = sess.run([global_step, loss, train_step])
            tf.logging.info('(%d) train loss: %f', step, train_loss)


if __name__ == '__main__':
    tf.app.run()
