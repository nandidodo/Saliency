
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



def classification_accuracy(preds, labels):

	N = len(preds)
	assert N==len(labels), 'Predictions and labels should be of same length'
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


