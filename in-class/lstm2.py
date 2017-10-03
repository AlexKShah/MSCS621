import numpy as np
import tensorflow as tf

#Example 2
sess = tf.Session()

LSTM_CELL_SIZE = 4  #4 hidden nodes = state_dim = the output_dim
input_dim = 6
num_layers = 2

cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
    cells.append(cell)

stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

#Batch size x time steps x features.
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input

#Run it
sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})

sess.close()
