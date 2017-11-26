# by Pablo Rivas
# vanilla script to view the weights using matplotlib

import numpy as np
import matplotlib.pyplot as plt

x = np.load('filters.npy')
p = np.zeros((2560,2560))

for i in range(10):
  for j in range(10):
    p[i*256:(i*256)+256,j*256:(j*256)+256] = np.reshape(x[:,(i*10)+j],(256,256))

plt.imshow(p,cmap='gray')
plt.colorbar()
plt.show()

