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
	
