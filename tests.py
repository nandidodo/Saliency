# this is where I just put really random things which are easy to find here. so we can inspect data and so forth. this isgoig to be a very freeform file. for scripting, as that's what python is meant to do.

import numpy as np
import scipy
from file_reader import *
from utils import *
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
"""
img = read_image(5)
#print img.shape
#show_colour_splits(img)

error_maps = load('error_map_test')
error_map = error_maps[1]
print error_map.shape
err = np.reshape(error_map, [28,28])
print err.shape
plt.imshow(err)
plt.show()"""

"""
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
print xtrain.shape

red = xtrain[:,:,:,0]
print red.shape
redimg = red[5]
print redimg.shape


plt.imshow(xtrain[5])
plt.show()

plt.imshow(redimg)
plt.show()
"""

a = [[1,5,3],[4,5,7],[7,5,2]]
a = np.array(a)
print a
print a.shape
print np.argmax(a)
print np.argmax(a, axis=0)
print np.argmax(a, axis=1)

# okay, fuck it. I'm going to search the entire array.this is going to take for fucking ever, and be totally terrible, but I'm going to do it, because I don't understnad what the fuck numpy argmax is actually doing?



