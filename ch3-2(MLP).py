# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

eta = 0.001
epochs = 50
batch_size = 128
display_step = 1

n_hidden1 = 512
n_hidden2 = 512

n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None, n_classes], name="y")

with tf.name_scope("layer1") as scope:
    w1 = tf.get_variable("w1", shape=[n_input, n_hidden1], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.fill([n_hidden1], 0.1), name="b1")
    z1 = tf.matmul(x, w1) + b1
    prob1 = tf.placeholder_with_default(1.0, shape=None, name="prob1")
    a1 = tf.nn.dropout(tf.nn.relu(z1), prob1)
    w1_summ = tf.summary.histogram("w1", w1)
    b1_summ = tf.summary.histogram("b1", b1)

with tf.name_scope("layer2") as scope:
    w2 = tf.get_variable("w2", shape=[n_hidden1, n_hidden2], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.fill([n_hidden2], 0.1), name="b2")
    z2 = tf.matmul(a1, w2) + b2
    prob2 = tf.placeholder_with_default(1.0, shape=None, name="prob2")
    a2 = tf.nn.dropout(tf.nn.relu(z2), prob2)
    w2_summ = tf.summary.histogram("w2", w2)
    b2_summ = tf.summary.histogram("b2", b2)

with tf.name_scope("layer3") as scope:
    w3 = tf.get_variable("w3", shape=[n_hidden2, n_classes], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.fill([n_classes], 0.1), name="b3")
    z3 = tf.matmul(a2, w3) + b3
    a3 = tf.nn.softmax(z3)
    w3_summ = tf.summary.histogram("w3", w3)
    b3_summ = tf.summary.histogram("b3", b3)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=eta).minimize(cost)
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("acc") as scope:
    pred = tf.argmax(a3, 1)
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
                                   feed_dict={x: batch_xs, y: batch_ys, prob1: 0.8, prob2: 0.8})
            avg_cost += cost_val/total_batch
        summary, train_accuracy = sess.run([merged, accuracy], {x: mnist.train.images, y: mnist.train.labels})
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
