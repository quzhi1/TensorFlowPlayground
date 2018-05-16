import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 28, 28, 3])
W = tf.Variable(tf.ones([4, 4, 3, 2]))
b = tf.Variable(tf.ones([2]))

stride_x = 1
stride_y = 1
m = tf.nn.conv2d(x, W, strides=[1, stride_x, stride_y, 1], padding='VALID') + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
z = sess.run(m, feed_dict={x: np.ones([5, 28, 28, 3])})
print(z.shape)
