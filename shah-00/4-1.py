# 4-1.py
# Alex Shah
# Homework 0

import numpy as np
import tensorflow as tf

x = tf.Variable(tf.float32)
y = tf.Variable(tf.float32)

y_eq = -3 * x ** 2 + 24 * x - 30

loss = tf.square (y - y_eq)
