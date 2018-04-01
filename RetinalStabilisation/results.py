#this is where the results are calculated before plotting
#not sure it is totaly needed to be honest, but could be a useful separation of concerns

import numpy as np 
import scipy
import keras
from keras.models import load_model
from utils import *
from plotting import *


def calculate_average_error(augments_name, copies_name, save_name=None):
	augment_test = np.load(augments_name + '_test.npy')
	augment_preds = np.load(augments_name+'_preds.npy')

	print augment_test.shape
	print augment_preds.shape

	copy_test = np.load(copies_name + '_test.npy')
	copy_preds = np.load(copies_name + '_preds.npy')

	print copy_test.shape
	print copy_preds.shape

	print np.max(augment_test)

	augment_errmaps = get_error_maps(augment_test, augment_preds)
	copy_errmaps = get_error_maps(copy_test, copy_preds)
	augment_error = get_mean_error(augment_errmaps)
	copy_error =get_mean_error(copy_errmaps)

	print "mnist augments error: " , augment_error
	print "mnist copies error: " , copy_error

	if save_name is not None:
		save_array([augment_error, copy_error], save_name)
		
	return augment_error, copy_error

	# the results work, it is much more successful with data augmentation
	# that is good!


#results: 
#mnist augments error:  0.0458422217494
#mnist copies error:  1.46164913036

#validation errors written down for the copies:
#in form epoch: train_loss, val_loss
#1: 0.0222, 0.0100
#2: 0.0124, 0.0083
#3: 0.0109, 0.0075
#4: 0.0100, 0.0070
# 5: 0.0094, 0.0067
# 6: 0.0089, 0.0064
#7@ 00086, 0.0062
#8: 0.0083, 0.0060
#9: 0.0081, 0.0058
#10: 0.0079, 0.0057

# not sure about the augments, I've either got to find this or reprint it



#I'll need something that will be able to calcualte the training curves
# from the histories
# and then obviously load the models and continue training
# so that it is possible to see the dissapearing prediction errors
#which is just the error maps
#which could be interesting!

# okay, aim to test here how things are

def save_history_losses(his_fname, save_fname):
	his = load(his_fname)
	history = his['history']
	his = None # free history
	loss = history['loss']
	val_loss = history['val_loss']
	print type(loss)
	print type(val_loss)
	np.save(save_fname+'_training_loss',loss)
	np.save(save_fname+'_validation_loss',val_loss)
	print "saved"
	return loss, val_loss


def test_results():
	aug_his = load('mnist_augments_history')
	val_data = aug_his['validation_data']
	#print len(val_data)
	#print val_data[0].shape
	print aug_his.keys()
	history = aug_his['history']
	print "history"
	print type(history)
	print len(history)
	print history.keys()
	loss = history['loss']
	val_loss = history['val_loss']
	print type(loss)
	print len(loss)
	print type(val_loss)
	print len(val_loss)
	params = aug_his['params']
	print "params"
	print type(params)
	print len(params)
	print params.keys()
	epoch = aug_his['epoch']
	print "epoch"
	print type(epoch)
	print len(epoch)
	print "validation data"
	print type(val_data)
	print len(val_data)


#and some quick tests
if __name__ == '__main__':
	print "In main!"
	#test_results()
	#calculate_average_error('mnist_augments', 'mnist_copies')
	#plot_errmaps('mnist_augments', 'mnist_copies')
	save_history_losses('mnist_augments_history', 'augments')
	save_history_losses('mnist_copies_history','copies')



