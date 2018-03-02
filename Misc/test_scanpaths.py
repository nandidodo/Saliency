import numpy as np
import cPickle as pickle
import scipy
import * from scanpaths

def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


#test loading the image
fname = "testsaliences_combined"
sals = load_array(fname)
print sals.shape
