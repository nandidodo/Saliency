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


def plot_errmaps(augments_name, copies_name, N =20):
	augment_test = np.load(augments_name + '_test.npy')
	augment_preds = np.load(augments_name+'_preds.npy')

	copy_test = np.load(copies_name + '_test.npy')
	copy_preds = np.load(copies_name + '_preds.npy')

	augment_errmaps = get_error_maps(augment_test, augment_preds)
	copy_errmaps = get_error_maps(copy_test, copy_preds)
	augment_error = get_mean_error(augment_errmaps)
	copy_error =get_mean_error(copy_errmaps)

	#reshape so work as images
	sh = augment_test.shape # all shapes should be the same here!
	augment_test = np.reshape(augment_test, (sh[0], sh[1],sh[2]))
	augment_preds= np.reshape(augment_preds, (sh[0], sh[1],sh[2]))
	augment_errmaps = np.reshape(augment_errmaps, (sh[0], sh[1],sh[2]))

	copy_test = np.reshape(copy_test, (sh[0], sh[1],sh[2]))
	copy_preds = np.reshape(copy_preds, (sh[0], sh[1],sh[2]))
	copy_errmaps = np.reshape(copy_errmaps, (sh[0], sh[1],sh[2]))

	for i in xrange(N):
		#begin the plot
		fig = plt.figure()

		ax1 = fig.add_subplot(131)
		plt.imshow(augment_test[i], cmap='gray')
		plt.title('Augmented Test image')
		plt.xticks([])
		plt.yticks([])

		ax2 = fig.add_subplot(132)
		plt.imshow(augment_preds[i], cmap='gray')
		plt.title('Augmented Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax3 = fig.add_subplot(133)
		plt.imshow(augment_errmaps[i], cmap='gray')
		plt.title('Augmented error map')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(231)
		plt.imshow(copy_test[i], cmap='gray')
		plt.title('Copies Test image')
		plt.xticks([])
		plt.yticks([])

		ax5 = fig.add_subplot(232)
		plt.imshow(copy_preds[i], cmap='gray')
		plt.title('Copies Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(233)
		plt.imshow(copy_errmaps[i], cmap='gray')
		plt.title('Copies error map')
		plt.xticks([])
		plt.yticks([])

		fig.tight_layout()
		plt.show()
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
def test_results():
	aug_his = load('mnist_augments_history')
	val_data = aug_his['validation_data']
	#print len(val_data)
	#print val_data[0].shape
	print aug_his.keys()
	hisory = aug_his['history']
	print "history"
	print type(history)
	print len(history)
	print history.keys()
	params = aug_his['params']
	print "params"
	print type(params)
	print len(params)
	epoch = aug_his['epoch']
	print "epoch"
	print type(epoch)
	print len(epoch)


#and some quick tests
if __name__ == '__main__':
	print "In main!"
	test_results()
	#calculate_average_error('mnist_augments', 'mnist_copies')
	#plot_errmaps('mnist_augments', 'mnist_copies')



