import numpy as np
import cPickle as pickle
import scipy
from scanpaths import *
import matplotlib.pyplot as plt
from utils import *


#test loading the image
fname = "testsaliences_combined"
img_fname = "testimages_combined"
imgs = load_array(fname)
print imgs.shape
sals = load_array(fname)
print sals.shape
#get sal
sal = sals[0]
sh = sal.shape
print sh
sal = sal[:,:,0]
print sal.shape
sal = np.reshape(sal, (sh[0], sh[1]))
print sal.shape

#path, scanpaths = simple_max_val_scanpath(sal, 5)
#plt.imshow(path)
#plt.show()


for i in xrange(10):
	img = imgs[i]
	img = img[:,:,0]
	img = np.reshape(img, (img.shape[0], img.shape[1]))
	sal = sals[i]
	sal = sal[:,:,0]	
	sal = np.reshape(sal, (sal.shape[0], sal.shape[1]))
	plot_nine_scanpaths(img, sal)
#simple path works

#this gets us sal
