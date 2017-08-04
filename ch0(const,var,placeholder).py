import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

g3 = tf.Graph()
with g3.as_default():
    v = tf.Variable(initial_value=1, name="v3")
    v = tf.assign(v, v+1)

with tf.Session(graph=g3) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v))
    print(sess.run(v))