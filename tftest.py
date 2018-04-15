# and gate weights {{{0, 2, 2}, {2, 0, -1}, {2, -1, 0}}}
#-----------------------------------------------------------------------
#INTRODUCTORY EXAMPLE
# import tensorflow as tf
#
# node1 = tf.placeholder(tf.float32)
# node2 = tf.placeholder(tf.float32)
# node3 = tf.add(node1, node2)
# node4 = node3 * 3
#
# sess = tf.Session()
#
# print(sess.run([node3, node4], {node1: [1, 3], node2: [4, 5]}))

#-----------------------------------------------------------------------
# #TENSORBOARD EXAMPLE
# import tensorflow as tf
#
# a = tf.constant(5, name = "input_a")
# b = tf.constant(3, name = "input_b")
# c = tf.multiply(a, b, name = "multiply_c")
# d = tf.add(a,b, name = "add_d")
# e = tf.add(c,d, name = "add_e")
#
# sess = tf.Session()
#
# output = sess.run(e)
# writer = tf.summary.FileWriter('./mygraph', sess.graph)
# #go to cmd, activate tensorflow
# #tensorboard --logdir= (insert path to the folder mygraph created with FileWriter)
# #copy and paste url into browser to open tensorboard
# writer.close()
# sess.close()
#-----------------------------------------------------------------------
# #TRAINING EXAMPLE
# import tensorflow as tf
#
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
#
# linear_model = W * x + b
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# #train to minimize our loss function
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# for i in range(1000):
#     sess.run(train, {x: [1,2,3,4], y:[0,-1,-2,-3]})
#
# print(sess.run([loss, W, b], {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#
# writer = tf.summary.FileWriter('./mygraph', sess.graph)
# writer.close()
# sess.close()
#-----------------------------------------------------------------------
from __future__ import print_function, division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.array(np.random.choice(3, 50, p = [.5,.2, .3]))
x = x.reshape((5, -1))
print(x)

















