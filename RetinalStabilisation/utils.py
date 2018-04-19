
#some simple utils for things
from __future__ import division
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from skimage import exposure
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, TerminateOnNaN, ReduceLROnPlateau
import os
from math import gamma


#okay, I need to do simple things to get medians and t-tests as that is fairly straightforward hopefully
#for exlporatory data analysis, and to have some statistical backing
def median(data):
	N = len(data)
	if N %2 ==0:
		#so even
		half = N//2
		d1 = data[half]
		d2 = data[half+1]
		return (d1 + d2)/2
	if N % 2 !=0:
		return data[(N//2)+1]

def variance(data):
	return np.var(data)

def t_test(d1, d2):
	N1 = len(d1)
	N2 = len(d2)
	mu1 = np.mean(d1)
	mu2 = np.mean(d2)
	var1 = np.var(d1)
	var2 = np.var(d2)
	numerator = np.abs(mu1-mu2)
	denom = np.sqrt((((N1+1)* var1 + (N2+1)*var2)/(N1+N2+2))*(1/N1+1) * (1/N2+1))
	t = numerator/denom
	#not sure precisely how to get degrees of freedom here!
	p = t_distribution_mine(t, N1-1)
	return t

	#I guess now I'm working on this... so, what is the ultimate plan here?
	# presumably to actually run a t-test on things, or at least determine variance

def t_distribution_mine(t, df):
	const = gamma((df+1)/2) / (np.sqrt(df * np.pi) * gamma(df/2))
	exp = (1 + np.square(t)/df)**(-1*(df+1)/2)
	return const*exp

def KL_divergence(p,q):
	#assumes p and q are lists of probability values which have been normalised
	# so it's the discrete kl, so not difficult to calculate
	if len(p) !=len(q):
		raise ValueError('The distributions must have matching lengths')

	total = 0
	for i in xrange(len(p)):
		total += p[i] + np.log(p[i]/q[i])
	return total

#stirlings approximation for calculating the gamma factor - i.e. factorial - an old method
# lanczos' method is better! but seems more complicated
def stirling_gamma_approximation(n):
	#define e.
	e = np.exp(-1)
	return np.sqrt(2*np.pi*n) * np.power((n/e), n)

def one_hot(labels,num_classes=None):
	if num_classes is None:
		#guess from the labels
		num_classes = np.max(labels) +1
	N = len(labels)
	#create our new array
	one_hot_array = np.zeros((N, num_classes))
	print "one hot array shape: " , one_hot_array.shape
	for i in xrange(N):
		one_hot_array[i][int(labels[i])] = 1
	return one_hot_array


def classification_accuracy(preds, labels):

	N = len(preds)
	assert N==len(labels), 'Predictions and labels should be of same length'
	assert preds.shape == labels.shape, 'Predictions and labels should have same shape'
	total = 0
	for i in xrange(N):
		if np.argmax(preds[i]) == np.argmax(labels[i]):
			total+=1
	percent = (total/N)*100
	return percent


def get_error_map(original_img, pred):
	assert original_img.shape ==pred.shape, 'Original image and prediction must have same shape'
	#for the moment just do a subtraction
	return np.abs(original_img - pred)

def get_total_error(error_map):
	return np.sum(error_map)


def get_error_maps(imgs, preds):
	print "In get error maps"
	print imgs.shape
	print preds.shape
	assert imgs.shape == preds.shape, 'Images and predictions must be the same shape'
	err_maps = []
	for i in xrange(len(imgs)):
		errmap = get_error_map(imgs[i], preds[i])
		err_maps.append(errmap)

	err_maps = np.array(err_maps)
	return err_maps


def get_mean_error(errmaps):
	print "In get mean error"
	print errmaps.shape
	N = len(errmaps)
	total = 0
	for i in xrange(N):
		total += get_total_error(errmaps[i])
		#print "bib"
	print "return value"
	print total/N
	return total/N

def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


def serialize_class_object(f):
	try:
		return dict((k,v) for k,v in f.__dict__.iteritems() if not callable(v) and not k.startswith('__'))
	except Exception as err:
		print "Exception in Serialization: " + str(err)
		return {"Error" : err}



def build_callbacks(save_path, min_delta = 1e-4, patience = 10, histogram_freq=0):
	
	checkpointer = ModelCheckpoint(filepath=os.path.join(save_path, "_weights"), monitor="val_loss",save_best_only=True, save_weights_only=True)
	
	early_stopper = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience*2)

	epoch_logger = CSVLogger(os.path.join(save_path, "epoch_logs.csv"))
	
	#batch_logger= BatchLossCSVLogger(os.path.join(save_path, "batch_logs.csv"))
	
	tensorboard = TensorBoard(log_dir=(os.path.join(save_path, '_tensorboard_logs')), histogram_freq=histogram_freq, write_grads=(histogram_freq>0))

	terminator = TerminateOnNaN()
	
	reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience= patience, verbose=1, mode='auto', min_lr = 1e-8)

	return [checkpointer, early_stopper, epoch_logger, tensorboard, terminator, reduceLR]


