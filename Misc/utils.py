import numpy as np
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from skimage import exposure
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, TerminateOnNaN, ReduceLROnPlateau
import os


def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


def split_into_test_train(data, frac_train = 0.9, frac_test = 0.1):
	assert frac_train + frac_test == 1, 'fractions must add up to one'
	length = len(data)
	#print length
	#print frac_train*length
	train = data[0:int(frac_train*length)]
	test = data[int(frac_train*length): length]
	return train, test


#get salmap - I'm not sure if this is the way to do it, or just a simple way, but it's what I'm using for now. I can alter it later without changing the other code!
def get_salmaps(imgs, preds):
	assert len(imgs.shape)==2 or len(imgs.shape)==3, 'Images must be two dimensional'
	assert len(preds.shape)==2 or len(preds.shape)==3, 'Preds must be two dimensional'
	assert imgs.shape==preds.shape, 'Preds and images must have same shape'
	if len(imgs.shape)==2:
		return imgs - preds
	
	#begin loop
	sal_maps = []
	for i in xrange(len(imgs)):
		#calculate the sal map by simple subtraction for now!
		sal_maps.append(imgs[i] - preds[i])

	sal_maps = np.array(sal_maps)
	return sal_maps


def serialize_class_object(f):
	try:
		return dict((k,v) for k,v in f.__dict__.iteritems() if not callable(v) and not k.startswith('__'))
	except Exception as err:
		print "Exception in Serialization: " + str(err)
		return {"Error" : err}



def build_callbacks(save_path, min_delta = 1e-4, patience = 10, histogram_freq=0):
	
	checkpoint_filepath = os.path.join(save_path,'checkpoint_{epoch:02d}-{val_loss:.2f}.hd5')
	checkpointer = ModelCheckpoint(checkpoint_filepath, monitor="val_loss",save_best_only=True, save_weights_only=True)
	#so it's only the best weights, but how do I use it!
	
	early_stopper = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience*2)

	epoch_logger = CSVLogger(os.path.join(save_path, "epoch_logs.csv"))
	
	#batch_logger= BatchLossCSVLogger(os.path.join(save_path, "batch_logs.csv"))
	
	#tensorboard = TensorBoard(log_dir=(os.path.join(save_path, '_tensorboard_logs')), histogram_freq=histogram_freq, write_grads=(histogram_freq>0))

	terminator = TerminateOnNaN()
	
	reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience= patience, verbose=1, mode='auto', min_lr = 1e-8)

	return [checkpointer, early_stopper, epoch_logger, terminator, reduceLR]
