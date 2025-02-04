import tensorflow as tf

#Start interactive session
sess = tf.InteractiveSession()
sess.close()
sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

# 1 channel image 
# 28x28
# -1 interpreted as pending/default - will be batch # later

x_image = tf.reshape(x, [-1,28,28,1])  
x_image

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs
# 32 different ways/filters of reading every number for one image

convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

h_conv1 = tf.nn.relu(convolve1)
# REctify Linear Unit
# "relu" if results less than 0, makes it 0, 
# makes it easier for computation

# max pooling specifies a size of which to get max of that region
# image 28x28
# max pool 2x2
# output = 14x14
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv1

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

h_conv2 = tf.nn.relu(convolve2)

conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2


# flatten second layer into 1024 x 1
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

# Fully Connected Layer

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

h_fc1 = tf.nn.relu(fcl)
h_fc1

# Dropout layer
# tried to inflict "amnesia" make some values 0 to ignore
# makes it more robust to noise or variations
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

# 1024 because that's the prev layer
# 10 for outputs, num of classes
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

fc=tf.matmul(layer_drop, W_fc2) + b_fc2

y_CNN= tf.nn.softmax(fc)
y_CNN

### Design End ###

# Training


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

# Another good optimizer, faster than grad desc
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# compare output to actual (highest 1)
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

################################################################################

#Baseline:

#for i in range(1000):
#    batch = mnist.train.next_batch(50)
#    if i%100 == 0:
#        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#baseline accuracy = 0.9651
#Time:
#real	0m9.208s
#user	0m9.724s
#sys	0m1.524s

################################################################################

## Hw 1
# Run full dataset

#for i in range(20000):
#    batch = mnist.train.next_batch(50)
#    if i%100 == 0:
#        train_accuracy = accuracy.eval(feed_dict={
#            x:batch[0], y_: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, train_accuracy))
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# full dataset accuracy = 0.9929
# Time:
#real	1m36.763s
#user	1m51.808s
#sys	0m15.312s

################################################################################

# Custom 1:

#for i in range(20000):
#    batch = mnist.train.next_batch(10)
#    if i%100 == 0:
#        train_accuracy = accuracy.eval(feed_dict={
#            x:batch[0], y_: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, train_accuracy))
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Custom 1 accuracy = 0.9894
#Time:
#real	1m7.127s
#user	1m25.648s
#sys	0m10.256s

################################################################################

# Custon 2:

#for i in range(20000):
#    batch = mnist.train.next_batch(100)
#    if i%100 == 0:
#        train_accuracy = accuracy.eval(feed_dict={
#            x:batch[0], y_: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, train_accuracy))
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Custom 2 accuracy = 0.9917
# Time:
#real	2m31.050s
#user	2m47.836s
#sys	0m23.852s


print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()
