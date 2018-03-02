import numpy as np
import cPickle as pickle
import scipy
from scanpaths import *

def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


#test loading the image
fname = "testsaliences_combined"
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

#this gets us sal
