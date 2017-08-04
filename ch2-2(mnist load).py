import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data

mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=False)
train, train_ans = mnist_images.train.next_batch(55000)
test, test_ans = mnist_images.test.next_batch(10000)
valid, valid_ans = mnist_images.validation.next_batch(5000)

print "list of values loaded", train_ans
example_to_visualize = 7
print "element N " + str(example_to_visualize + 1) + " of the list plotted"

image = train[example_to_visualize,:]
image = np.reshape(image, [28,28])
plt.imshow(image)
plt.show()