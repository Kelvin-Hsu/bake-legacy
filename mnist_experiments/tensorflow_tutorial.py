import tensorflow as tf
import numpy as np


def create_mnist_data():

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Load the training data
    x_train = mnist.train.images
    y_train = mnist.train.labels

    # Load the validation data
    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    # Add the validation data to the training data
    x = np.concatenate((x_train, x_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)

    # Load the testing data
    x_test = mnist.test.images
    y_test = mnist.test.labels

    return x, y, x_test, y_test


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


with tf.name_scope('model'):

    tf.set_random_seed(0)

    with tf.name_scope('data'):

        x = tf.placeholder(tf.float32, shape=[None, 784])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('convolutional_layer_1'):

        w_conv_1 = weight_variable([5, 5, 1, 32])
        b_conv_1 = bias_variable([32])

        h_conv_1 = tf.nn.relu(conv2d(x_image, w_conv_1) + b_conv_1)
        h_pool_1 = max_pool_2x2(h_conv_1)

        tf.summary.histogram('w_conv_1', w_conv_1)
        tf.summary.histogram('b_conv_1', b_conv_1)

    with tf.name_scope('convolutional_layer_2'):

        w_conv_2 = weight_variable([5, 5, 32, 64])
        b_conv_2 = bias_variable([64])

        h_conv_2 = tf.nn.relu(conv2d(h_pool_1, w_conv_2) + b_conv_2)
        h_pool_2 = max_pool_2x2(h_conv_2)

    with tf.name_scope('fully_connected_layer'):

        w_fc_1 = weight_variable([7 * 7 * 64, 1024])
        b_fc_1 = bias_variable([1024])

        h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
        h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, w_fc_1) + b_fc_1)

    with tf.name_scope('dropout'):

        keep_prob = tf.placeholder(tf.float32)
        h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob)

    with tf.name_scope('softmax_regression_layer'):

        w_fc_2 = weight_variable([1024, 10])
        b_fc_2 = bias_variable([10])
        y_conv = tf.matmul(h_fc_1_drop, w_fc_2) + b_fc_2

    with tf.name_scope('cross_entropy'):

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

    with tf.name_scope('accuracy'):

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('training'):

        gradients = tf.gradients(cross_entropy,
                                 [w_conv_1, b_conv_1, w_conv_2, b_conv_2,
                                  w_fc_1, b_fc_1, w_fc_2, b_fc_2])
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sequential_batch = True
n_sgd_batch = 50
dropout = 0.5
directory = './mnist_tutorial/'

x_train, y_train, x_test, y_test = create_mnist_data()

n = x_train.shape[0]

test_feed_dict = {x: x_test, y: y_test, keep_prob: 1.0}

if sequential_batch:

    epoch = 0
    perm_indices = np.random.permutation(np.arange(n))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):

    if sequential_batch:

        if i * n_sgd_batch % n > epoch:
            epoch = i * n_sgd_batch % n
            perm_indices = np.random.permutation(np.arange(n))

        sgd_indices = np.arange(i * n_sgd_batch, (i + 1) * n_sgd_batch) % n
        x_batch = x_train[perm_indices][sgd_indices]
        y_batch = y_train[perm_indices][sgd_indices]

    else:

        sgd_indices = np.random.choice(n, n_sgd_batch, replace=False)
        x_batch = x_train[sgd_indices]
        y_batch = y_train[sgd_indices]

    batch_train_feed_dict = {x: x_batch, y: y_batch, keep_prob: dropout}

    if i % 1 == 0:

        np.savez('%sparameter_info_%d.npz' % (directory, i),
                 w_conv_1=sess.run(w_conv_1),
                 b_conv_1=sess.run(b_conv_1),
                 w_conv_2=sess.run(w_conv_2),
                 b_conv_2=sess.run(b_conv_2),
                 w_fc_1=sess.run(w_fc_1),
                 b_fc_1=sess.run(b_fc_1))

        batch_train_accuracy = \
            sess.run(accuracy, feed_dict=batch_train_feed_dict)
        test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
        batch_train_cross_entropy = \
            sess.run(cross_entropy, feed_dict=batch_train_feed_dict)
        test_cross_entropy = sess.run(cross_entropy, feed_dict=test_feed_dict)
        grad = sess.run(gradients, feed_dict=batch_train_feed_dict)
        grad_norms = np.array([np.max(np.abs(grad_i)) for grad_i in grad])

        print("Step %d: "
              "Training accuracy: %g, "
              "Test accuracy: %g, "
              "Training Cross Entropy: %g, "
              "Test Cross Entropy: %g"
              % (i,
                 batch_train_accuracy, test_accuracy,
                 batch_train_cross_entropy, test_cross_entropy))
        print('Gradient Norms: ', grad_norms)

        np.savez('%strain_test_info_%d.npz' % (directory, i),
                 batch_train_acc=batch_train_accuracy,
                 batch_train_cel=batch_train_cross_entropy,
                 test_acc=test_accuracy,
                 test_cel=test_cross_entropy,
                 grad_norms=grad_norms)

    sess.run(train_step, feed_dict=batch_train_feed_dict)
