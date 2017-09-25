# 4-3.py
# Alex Shah
# Homework 0

import tensorflow as tf
import numpy as np

# Matrices
a = np.array([[1.0,4,3], [2,-1,3]])
b = np.array([[-2,0,5], [0,-1,4]])
c = np.array([[1,0], [0,2]])

# tf const
a = tf.constant(a.astype(float))
b = tf.constant(b.astype(float))
c = tf.constant(c.astype(float))


# matrix cannot be multiplied as is, 
# inner dimensions are different,
# transpose in order to multiply    
At = tf.transpose(a)
Bt = tf.transpose(b)
Ci = tf.matrix_inverse(c)

with tf.Session() as sess:
    
    try: print(tf.matmul(a,b).eval())
    except Exception as err:
        print("\n A x B fails as dimensions are different:")
        print(err)
    
    print("\n At x B")
    res = tf.matmul(At,b).eval()
    print(res)
    
    print("\n Rank")
    print(np.linalg.matrix_rank(res))
    
    print("\n A x Bt")
    res2 = tf.matmul(a,Bt).eval()
    print(res2)
    
    print("\n A x Bt + C inverse")
    print(tf.add(res2, Ci).eval())
