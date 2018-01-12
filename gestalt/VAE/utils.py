#various util functions

import numpy as np
import cPickle as pickle
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, TerminateOnNaN, ReduceLROnPlateau
import os

def get_run_num():
	if len(sys.argv)>1:
		return sys.argv[1]

#pickle loading and saving functoinality

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



def split_first_test_train(data, frac_train = 0.9):
	assert frac_train <=1, "frac_train must be a fraction"
	frac_test = 1-frac_train
	N = len(data)
	train = data[int(frac_test*N):N]
	test = data[0:int(frac_test*N)]
	return train, test

def split_first_test_val_train(data, frac_train =0.9, frac_val = 0.05, frac_test = 0.05):
	assert frac_train + frac_val + frac_test ==1, "train test validation splits must add up to one"
	N = len(data)
	len_test = int(frac_test*N)
	len_val = int(frac_val*N)
	#len_train = int(frac_train *N)
	test = data[0:len_test]
	val = data[len_test: (len_test+len_val)]
	train =data[(len_test+len_val):N]
	return train, val, test 


	#we do this with three dimensions at first, I think, because there's no particular reason to do it with less as we're trying to imitate across the cortex
	#I think horizontal is just the thing
	#we assume all images in the dataset are cropped to the same width - a big assumption, so we need that preprocessing step for this to work really
	img_width = len(dataset[0])
	shape = dataset.shape
	print shape
	half= img_width/2
	if len(shape) ==4: # i.e a 3d dimensioanl image
		leftsplit = dataset[:,:,0:half,:]
		rightsplit = dataset[:,:, half:img_width,:]
		leftslice = dataset[:,:,half-split_width:half,:]
		rightslice = dataset[:,:,half: half+split_width,:]
		return leftsplit, rightsplit, leftslice, rightslice

	if len(shape)==3: # i.e. a 3d image
		leftsplit = dataset[:,:,0:half]
		rightsplit = dataset[:,:, half:img_width]
		leftslice = dataset[:,:,half-split_width:half]
		rightslice = dataset[:,:,half: half+split_width]
		return leftsplit, rightsplit, leftslice, rightslice

def split_dataset_center_slice(dataset, split_width):

	#this just returns the equivalent of the leftslpit and the rightsplit
	#also assumes three dimensional images and four d dataset
	#also all images are cropped to the same width
	shape = dataset.shape
	
	img_width = len(dataset[0])
	half = img_width/2
	if len(shape) ==4:
		leftslice = dataset[:,:,half-split_width:half,:]
		rightslice = dataset[:,:,half: half+split_width,:]
		return leftslice, rightslice
	if len(shape)==3:
		leftslice = dataset[:,:,half-split_width:half]
		rightslice = dataset[:,:,half: half+split_width]
		return leftslice, rightslice
	


