import os

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# Note 1: this is a single layer softmax model
# Note 2: If you can't download the mnist, run: /Applications/Python 3.6/Install Certificates.command

# Disable GPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load mnist data set (train, validation, test)
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

# Create weights and biases
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([1, 10]))

# Softmax regression and cross entropy as cost function
yHat = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(yHat), axis=1))
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(yHat, 1))  # find which number is predicted
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Run the algorithm
for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(300)
    sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
    if _ % 20 == 0:
        print('step ', _, sess.run([cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys}))
