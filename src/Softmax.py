import os

import tensorflow as tf

# Note: this is a single layer softmax model

# Disable GPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load data
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Placeholder for X and Y
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

# Variable for W and b
W = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([1, 3]))

# Prediction, Loss function, Gradiant
yHat = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(yHat), axis=1))
# cost = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=yHat)  # TODO Why doesn't this work
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for steps in range(1000):
        predict = sess.run(yHat, feed_dict={X: x_data, Y: y_data})
        print(steps, predict)
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        cost_per_step = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        print(steps, cost_per_step)
