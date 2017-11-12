# Galaxy Image Processing
# Alex Shah
# 9/25/17

import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

# Galaxy Data
train_imgs = tf.constant(['train/img1.jpg', 'train/img2.jpg', 'train/img4.jpg', 'train/img5.jpg'])
train_labels = tf.constant([0,0,1,1])

test_imgs = tf.constant(['test/img3.jpg', 'test/img6.jpg'])
test_labels = tf.constant([0,1])

# TF Dataset Object
# "From slices" is a utility to slice dataset to create a tensor ready to be sliced (for a large dataset)
tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
te_data = Dataset.from_tensor_slices((test_imgs, test_labels))

# Two classes
NUM_CLASSES = 2

def input_parser(img_path, label):
  # Convert label to one hot encoding (important for probabilities)
  one_hot = tf.one_hot(label, NUM_CLASSES)
  
  # Read image from disk in binary, decode it
  img_file = tf.read_file(img_path)
  img_decoded = tf.image.decode_image(img_file,channels=3)
  ## put tensorflow method to resize here
  ###
  return img_decoded, one_hot
  
tr_data = tr_data.map(input_parser)
te_data = te_data.map(input_parser)

#TF Iterator Object
iterator = Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)

# Iterator enables us to get next element
next_element = iterator.get_next()

# Initialization
training_init_op = iterator.make_initializer(tr_data)
testing_init_op = iterator.make_initializer(te_data)

with tf.Session() as sess:
  # Initialize iterator on training data
  sess.run(training_init_op)
  # Get each element of training dataset until end reached
  while True:
    try:
      elem = sess.run(next_element)
	  # Image read process goes here:
	  ###
      print(elem)
    except tf.errors.OutOfRangeError:
      print("End of Data")
      break
  # Initialize iterator on test data
  sess.run(testing_init_op)
  while True:
    try:
      elem = sess.run(next_element)
	  # Image read process goes here:
	  ###
      print(elem)
    except tf.errors.OutOfRangeError:
      print("End of Data")
      break