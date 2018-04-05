#this is where the results are calculated before plotting
#not sure it is totaly needed to be honest, but could be a useful separation of concerns

from __future__ import division
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


def test_fixations():
	copy_results = load('results/from_scratch_fixation_copy')
	aug_results =load('results/from_scratch_fixation_augments')
	print type(copy_results)
	print len(copy_results)
	plot_fixation_errmaps(aug_results, copy_results)


def get_mean_total_error(errmaps):
	N = len(errmaps)
	total = 0
	for errmap in errmaps:
		total += np.sum(errmap)
	return total/N


def test_generative_invariance(aug_model, copy_model, results_save=None):
	# this tests the ivnariance of the model - i.e. how good it is at predicting
	# the whole image it is given for the different invairances tested#
	# I should also do a classificatory invariance as well, althoguh it would mean
	# training an entirely separate invariant classification, so that could work well too!

	#load the models
	aug_model = load_model(aug_model)
	copy_model = load_model(copy_model)

	#load the invariance files
	mnist_0px = np.load('data/mnist_invariance_0pixels_translate.npy')
	mnist_2px = np.load('data/mnist_invariance_2pixels_translate.npy')
	mnist_4px = np.load('data/mnist_invariance_4pixels_translate.npy')
	mnist_6px = np.load('data/mnist_invariance_6pixels_translate.npy')
	mnist_8px = np.load('data/mnist_invariance_8pixels_translate.npy')
	print mnist_0px.shape
	print mnist_2px.shape

	pixels = [0,2,4,6,8]

	invariances = [mnist_0px,mnist_2px,mnist_4px,mnist_6px,mnist_8px]

	aug_errors = []
	copy_errors = []

	#test on each and get errors
	for invariance in invariances:
		#reshape
		sh = invariance.shape
		invariance = np.reshape(invariance, (sh[0], sh[1], sh[2],1))

		aug_preds = aug_model.predict(invariance)
		copy_preds = copy_model.predict(invariance)
		aug_errmaps = get_error_maps(invariance, aug_preds)
		copy_errmaps = get_error_maps(invariance, copy_preds)
		aug_mean_error = get_mean_total_error(aug_errmaps)
		copy_mean_error = get_mean_total_error(copy_errmaps)
		aug_errors.append(aug_mean_error)
		copy_errors.append(copy_mean_error)

	aug_errors = np.array(aug_errors)
	copy_errors = np.array(copy_errors)

	if results_save:
		np.save(results_save+'_aug', aug_errors)
		np.save(results_save+'_copy', copy_errors)
		np.save(results_save+'_pixels', pixels)
	return aug_errors, copy_errors,pixels

def test_discriminative_invariance(aug_model, copy_model, results_save=None):
	aug_model = load_model(aug_model)
	copy_model = load_model(copy_model)

	mnist_0px_data = np.load('data/discriminative_0pixels_translate_data.npy')
	mnist_2px_data = np.load('data/discriminative_2pixels_translate_data.npy')
	mnist_4px_data = np.load('data/discriminative_4pixels_translate_data.npy')
	mnist_6px_data = np.load('data/discriminative_6pixels_translate_data.npy')
	mnist_8px_data = np.load('data/discriminative_8pixels_translate_data.npy')

	#I'm going to convert these to one-hot - do that here
	mnist_0px_labels = np.load('data/discriminative_0pixels_translate_labels.npy')
	mnist_2px_labels = np.load('data/discriminative_2pixels_translate_labels.npy')
	mnist_4px_labels = np.load('data/discriminative_4pixels_translate_labels.npy')
	mnist_6px_labels = np.load('data/discriminative_6pixels_translate_labels.npy')
	mnist_8px_labels = np.load('data/discriminative_8pixels_translate_labels.npy')

	pixels = [0,2,4,6,8]

	invariance_data = [mnist_0px_data,mnist_2px_data,mnist_4px_data,mnist_6px_data,mnist_8px_data]
	invariance_labels = [mnist_0px_labels,mnist_2px_labels,mnist_4px_labels,mnist_6px_labels,mnist_8px_labels]

	aug_accuracies = []
	copy_accuracies = []

	for i in range(len(invariance_data)):
		data = invariance_data[i]
		#reshape
		sh = data.shape
		data = np.reshape(data, (sh[0],sh[1],sh[2],1))

		#reshape labels
		labels = one_hot(invariance_labels[i])
		#predict
		aug_pred_labels = aug_model.predict(data)
		copy_pred_labels = copy_model.predict(data)
		print "predictions"
		print aug_pred_labels.shape
		#just get teh accuracies and save it
		aug_acc = classification_accuracy(labels, aug_pred_labels)
		copy_acc = classification_accuracy(labels, copy_pred_labels)

		print "pixels: " +str(pixels[i])
		print "aug acc: " , aug_acc
		print "copy acc: " , copy_acc

		aug_accuracies.append(aug_acc)
		copy_accuracies.append(copy_acc)

	aug_accuracies = np.array(aug_accuracies)
	copy_accuracies = np.array(copy_accuracies)
	# so this is quite strange, the accuracies here are just terrible
	# I wonder if the network can actuallylearn anything useful here
	# at least the pattern works. I think I'll need to learn it with like
	#50 epochs instead see if it helps hopefully!

	if results_save is not None:
		np.save(results_save+'_aug_accuracies', aug_accuracies)
		np.save(results_save+'_copy_accuracies', copy_accuracies)
		np.save(results_save+'_pixels', pixels)
	return aug_accuracies, copy_accuracies, pixels



#and some quick tests
if __name__ == '__main__':
	
	#print "In main!"
	#test_results()
	#calculate_average_error('mnist_augments', 'mnist_copies', save_name="errors_1")
	#plot_errmaps('mnist_augments', 'mnist_copies')
	#save_history_losses('mnist_augments_history', 'augments')
	#save_history_losses('mnist_copies_history','copies')

	#test_fixations()
	#test_generative_invariance('model_mnist_augments', 'model_mnist_copy','results/generative_invariance')
	test_discriminative_invariance('discriminative_aug_model','discriminative_copy_model', 'results/discriminative_invariance')
	# it sort of shows waht I want to show, but not that well, dagnabbit!

	# so for some reason, even though the validation and test errors are barely different
	# this is not the case for the error maps where there is a significantand consistent difference
	# in exactly the directoin I want, whic his good. Now that's the main results I need
	# all I will need to do then is to show it's superiority on a standard not reconstruction test
	# but on actual classification tasks, which should be easy, and interesting
	# as well as greater difference there
	# and also show the dissapearance of the error map - i.e. retinal stabilisation over time
	# so hopefully that should be fairly straightforward and with those results, I can begin a proper writeup
	# and have that to richard by the end of this week, which could be interesting#
	# so yeah, that would be good, and test classification

	# oh yes, with the new thing it actually dissapears over tiem
	# that's really fantastic, I just need some good way to present them to Richard and for the paper
	# the other results work out vaguely okay, I think, so that is nice,
	# I also obviously need the graphs of somethign else, so that is cool also,
	#yay!



