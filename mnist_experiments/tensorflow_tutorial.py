import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

with tf.name_scope('tutorial'):

    tf.set_random_seed(0)

    with tf.name_scope('training_data'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1,28,28,1])

    with tf.name_scope('convolutional_layer_1'):

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        tf.summary.histogram('W_conv1', W_conv1)
        tf.summary.histogram('b_conv1', b_conv1)

        # for i in range(32):
        #     h_conv1_i = tf.reshape(h_conv1[:, :, :, i], shape=[-1, 28, 28, 1], name='h_conv1_%d' % i)
        #     tf.summary.image('h_conv1_%d' % i, h_conv1_i, max_outputs=3)
        #     h_pool1_i = tf.reshape(h_pool1[:, :, :, i], shape=[-1, 14, 14, 1], name='h_pool1_%d' % i)
        #     tf.summary.image('h_pool1_%d' % i, h_pool1_i, max_outputs=3)

    with tf.name_scope('convolutional_layer_2'):

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # for i in range(64):
        #     h_conv2_i = tf.reshape(h_conv2[:, :, :, i], shape=[-1, 14, 14, 1], name='h_conv2_%d' % i)
        #     tf.summary.image('h_conv2_%d' % i, h_conv2_i, max_outputs=3)
        #     h_pool2_i = tf.reshape(h_pool2[:, :, :, i], shape=[-1, 7, 7, 1], name='h_pool2_%d' % i)
        #     tf.summary.image('h_pool2_%d' % i, h_pool2_i, max_outputs=3)

    with tf.name_scope('fully_connected_layer'):

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('softmax_regression_layer'):

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('cross_entropy'):

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('training'):

        gradients = tf.gradients(cross_entropy, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# # Start Training
# merged_summary = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('./mnist_tutorial/train/')
# train_writer.add_graph(sess.graph)
# test_writer = tf.summary.FileWriter('./mnist_tutorial/test/')
# test_writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())

import numpy as np

print(mnist.train.images.shape)

for i in range(20000):

  print('Step %d' % i)
  # batch = mnist.train.next_batch(100)
  # sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


  sgd_indices = np.random.choice(mnist.train.images.shape[0], 1000, replace=False)
  train_feed_dict = {x: mnist.train.images[sgd_indices], y_: mnist.train.labels[sgd_indices], keep_prob: 0.5}
  print('Obtained Batch')
  sess.run(train_step, feed_dict=train_feed_dict)
  grad = sess.run(gradients, feed_dict=train_feed_dict)

  print('Gradient Norms:', np.array([np.max(np.abs(grad_i)) for grad_i in grad]))

  if i % 100 == 0:
    # train_feed_dict = {x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}
    test_feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
    train_accuracy = sess.run(accuracy, feed_dict=train_feed_dict)
    test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
    train_cross_entropy = sess.run(cross_entropy, feed_dict=train_feed_dict)
    test_cross_entropy = sess.run(cross_entropy, feed_dict=test_feed_dict)
    grad = sess.run(gradients, feed_dict=train_feed_dict)

    print("Step %d: Training accuracy: %g, Test accuracy: %g, Training Cross Entropy: %g, Test Cross Entropy: %g" % (i, train_accuracy, test_accuracy, train_cross_entropy, test_cross_entropy))
    print('Gradient Norms: ', np.array([np.max(np.abs(grad_i)) for grad_i in grad]))

    # train_summary = sess.run(merged_summary, feed_dict=train_feed_dict)
    # train_writer.add_summary(train_summary, i)
    # test_summary = sess.run(merged_summary, feed_dict=test_feed_dict)
    # test_writer.add_summary(test_summary, i)



