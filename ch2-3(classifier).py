import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=False)

train, train_ans = mnist_images.train.next_batch(100)
test, test_ans = mnist_images.test.next_batch(10)

train_tensor = tf.placeholder("float", [None, 784])
test_tensor = tf.placeholder("float", [784])

distance = tf.sqrt(tf.reduce_sum(tf.square(train_tensor - test_tensor), reduction_indices=1))
pred = tf.arg_min(distance, 0)
accuracy = 0


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(len(test_ans)):
        nn_idx = sess.run(pred, feed_dict = {train_tensor: train, test_tensor: test[i, ]})
        print "Test N ", i, "Predicted Class: ", train_ans[nn_idx], "True Class: ", test_ans[i]
        if train_ans[nn_idx] == test_ans[i]:
            accuracy += 1.0/len(test_ans)
        else:
            image = np.reshape(train[nn_idx], [28, 28])
            plt.imshow(image)
            plt.show()
            image = np.reshape(test[i], [28, 28])
            plt.imshow(image)
            plt.show()
    print "Result = ", accuracy
