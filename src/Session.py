import os

import numpy as np
import tensorflow as tf

# Disable GPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Build the graph
W = tf.Variable(tf.random_normal([4, 2]))
X = tf.placeholder(tf.float32, shape=[None, 4])  # This is just a placeholder for input
b = tf.constant(2.0)
Z = tf.matmul(X, W) + b

# Open session
sess = tf.Session()
sess.run(W.initializer)  # Note, if initializer is not called, normal distribution won't happen
sess.run(Z, feed_dict={X: np.ones([2, 4])})

# Explore operations
print(tf.get_default_graph().get_operations())
mul_op = tf.get_default_graph().get_operation_by_name("MatMul")
print(mul_op)
add_op = tf.get_default_graph().get_operation_by_name("add")
print(add_op)

# Build in another way
y = tf.add(
    tf.multiply(
        tf.Variable(2, name="m"),
        tf.placeholder(tf.int32, shape=1, name="x")
    ), tf.Variable(1, name="b"))
