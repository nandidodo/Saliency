# okay, this is just for generic utils. such as saving functionality

import numpy as np
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle


#pickle loading and saving functoinality

def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def show_colour_splits(img, show_original = True):
	#assumes img is 4d so we can split along the colour	
	if show_original:
		print "ORIGINAL:"
		plt.imshow(img)
		plt.show()
	print "RED:"
	plt.imshow(img[:,:,0])
	plt.show()
	print "GREEN:"
	plt.imshow(img[:,:,1])
	plt.show()
	print "BLUE:"
	plt.imshow(img[:,:,2])
	plt.show()


def index_distance(indices1, indices2):
	#this finds the euclidian distance. we probably don't realy want any other type tbh
	assert len(indices1) == len(indices2),'indices must have same dimension'
	total = 0
	for i in xrange(len(indices1):
		total += (indices1[i] - indices2[i]) **2
	return np.sqrt(total)
		
	
