# -*- coding: utf-8 -*-
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

eta = 0.001
epochs = 50
batch_size = 128
display_step = 1

n_input = 784
img_size = [28, 28]
ch_color = 1
n_classes = 10

n_hidden = 1024
kernel_size = [3, 3]
n_filters = [32, 64]
dropout_prob = 0.8

x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None, n_classes], name="y")

x_image = tf.reshape(x, [-1, img_size[0], img_size[1], ch_color])

with tf.name_scope("conv1") as scope:
    W_conv1 = tf.get_variable("W_conv1",
                              shape=[kernel_size[0], kernel_size[1], ch_color, n_filters[0]],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.Variable(tf.fill([n_filters[0]], 0.1), name="b_conv1")
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("conv2") as scope:
    W_conv2 = tf.get_variable("W_conv2",
                              shape=[kernel_size[0], kernel_size[1], n_filters[0], n_filters[1]],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.Variable(tf.fill([n_filters[1]], 0.1), name="b_conv1")
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("fc1") as scope:
    h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2.shape[1].value * h_pool2.shape[2].value * n_filters[1]])
    W_fc1 = tf.get_variable("W_fc1",
                              shape=[h_pool2_flat.shape[1].value, n_hidden],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.fill([n_hidden], 0.1), name="b_fc1")
    prob = tf.placeholder_with_default(1.0, shape=None, name="prob")
    h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1), prob)

with tf.name_scope("fc2") as scope:
    W_fc2 = tf.get_variable("W_fc2",
                            shape=[n_hidden, n_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.Variable(tf.fill([n_classes], 0.1), name="b_fc1")
    z_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
    a_fc2 = tf.nn.softmax(z_fc2)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_fc2, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=eta).minimize(cost)
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("acc") as scope:
    pred = tf.argmax(z_fc2, 1)
    answ = tf.argmax(y, 1)
    correct_prediction = tf.equal(pred, answ)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc_summ = tf.summary.scalar("accuracy", accuracy)
    pred_summ = tf.summary.histogram("pred", pred)
    answ_summ = tf.summary.histogram("answ", answ)

with tf.Session() as sess:
    # tensorboard block
    tensorboard_writer = tf.summary.FileWriter("./logs/", sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={x: batch_xs, y: batch_ys, prob: dropout_prob})
            avg_cost += cost_val/total_batch
        summary, train_accuracy = sess.run([merged, accuracy], {x: mnist.train.images[0:10000], y: mnist.train.labels[0:10000]})
        if epoch % display_step == 0:
            print "Epoch ", '%04d' % (epoch + 1),\
                "cost: ", "{:.4f}".format(avg_cost),\
                "acc: ", "{:.4f}".format(train_accuracy)
        # tensorboard block
        tensorboard_writer.add_summary(summary, epoch)
    print "Training Phase Finished"
    test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print "MODEL accuracy : ", test_accuracy
    tensorboard_writer.close()