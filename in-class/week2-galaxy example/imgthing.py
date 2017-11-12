# Galaxy Image Processing 2
# Alex Shah
# 9/25/17

import tensorflow as tf

# Two classes
NUM_CLASSES = 2

def input_parser(img_path, label)
  # Convert label to one hot encoding (important for probabilities)
  one_hot = tf.one_hot(label, NUM_CLASSES)
  
  # Read image from disk in binary, decode it
  img_file = tf.read_file(img_path)
  img_decoded = tf.image.decode_image(img_file,channels=3)
  
  return img_decoded, one_hot
