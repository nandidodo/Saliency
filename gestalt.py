# so the aim of this is to do some brief work on gestalts, i.e. for our model we would then split an image in half and try to show some standard gestalt effects from this, whcih seems very reasonable indeed to be honest, but I honestly do not even knwo. we're probably going to majorly focus on this after the first paper is over, but we've got this to do first, so let's look at it

from keras.datasets import cifar10, mnist
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras.models import Model
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from file_reader import *
from utils import *
from autoencoder import *
from experiments import *
import sys

#get our arg number
run_num = sys.argv[1]
# we can then use this globally for the thing when we do our hyperparam grid search


def split_dataset_half_small_section(dataset,split_width):
	#we do this with three dimensions at first, I think, because there's no particular reason to do it with less as we're trying to imitate across the cortex
	#I think horizontal is just the thing
	#we assume all images in the dataset are cropped to the same width - a big assumption, so we need that preprocessing step for this to work really
	img_width = len(dataset[0])
	leftsplit = dataset[:,0:img_width/2,:,:]
	rightsplit = dataset[:,(img_width/2):img_width, :,:]
	leftslice = dataset[:,(img_width-split_width):img_width,:,:]
	rightslice = dataset[:,img_width:(img_width+split_width),:,:]
	return leftsplit, rightsplit, leftslice, rightslice

def split_dataset_center_slice(dataset, split_width):

	#this just returns the equivalent of the leftslpit and the rightsplit
	#also assumes three dimensional images and four d dataset
	#also all images are cropped to the same width

	img_width = len(dataset[0])
	leftslice = dataset[:,(img_width-split_width):img_width,:,:]
	rightslice = dataset[:,img_width:(img_width+split_width),:,:]
	return leftslice, rightslice
	


def plot_slice_splits(leftsplit, rightslpit, leftslice, rightslice, show=True):	
	fig = plt.figure()

	#originalcolour
	ax1 = fig.add_subplot(221)
	plt.imshow(leftsplit)
	plt.title('Left Half')
	plt.xticks([])
	plt.yticks([])

	#red
	ax2 = fig.add_subplot(222)
	plt.imshow(leftslice)
	plt.title('Left Slice')
	plt.xticks([])
	plt.yticks([])

	#green
	ax3 = fig.add_subplot(223)
	plt.imshow(rightsplit)
	plt.title('Right Half')
	plt.xticks([])
	plt.yticks([])

	##blue
	ax4 = fig.add_subplot(224)
	plt.imshow(rightslice)
	plt.title('Right Slice')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	if show:
		plt.show(fig)
	return fig


def show_split_dataset_with_slices(dataset, split_width):
	leftsplits, rightsplits, leftslices, rightslices = split_dataset_half_small_section(dataset, split_width=split_width)
	N = len(dataset)
	for i in xrange(N):
		plot_slice_splits(leftsplits[i], rightsplits[i], leftslices[i], rightslices[i])
	


def split_half_image_experiments_from_file(fname, epochs=100, save=True, test_up_to=None, preview=False, verbose=False, param_name=None, param=None, save_name=None, test_all=False):
	#we'll do this in the three dimensional test
	imgs = load_array(fname)
	train, test = imgs
	halftrain1, halftrain2 = split_image_dataset_into_halves(train)
	halftest1, halftest2 = split_image_dataset_into_halves(test)
	
	#okay, now we've got our train and test we run the actual experiment
	if preview:
		for i in xrange(10):
			compare_two_images(halftrain1[i], halftrain2[i], reshape=True)

	if param_name is None or param is None:
		a1 = Hemisphere(halftrain1, halftrain2, halftrain1, halftrain2)
	if param_name is not None and param is not None:
		a1 = Hemisphere(halftrain2, halftrain1, halftest2, halftest1, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(halftrain1, halftrain2,halftest1, halftest2)
	if param_name is not None and param is not None:
		a2 = Hemisphere(halftrain2, halftrain1, halftest2, halftest1, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	if test_all:
		a1=Hemisphere(halftrain1, halftrain2, halftrain1, halftrain2)
		a2=Hemisphere(halftrain2,halftrain1,halftrain2,halftrain1)
	

	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'spfreq_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + 'spfreq_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		if save_name is None:
			save_array(mean_maps, 'split_half_gestalt_mean_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return mean_maps


def split_predict_slice_from_half_from_file((fname,slice_pix=30, epochs=100, save=True, test_up_to=None, preview=False, verbose=False, param_name=None, param=None, save_name=None, test_all=False):
	#we'll do this in the three dimensional test
	imgs = load_array(fname)
	train, test = imgs
	lefthalftrain, righthalftrain, leftslicetrain, rightslicetrain = split_dataset_half_small_section(train, slice_pix)
	lefthalftest, righthalftest, leftslicetest, rightslicetest = split_dataset_half_small_section(test, slice_pix)

	if param_name is None or param is None:
		a1 = Hemisphere(lefthalftrain, rightslicetrain, lefthalftest, rightslicetest)
	if param_name is not None and param is not None:
		a1 = Hemisphere(lefthalftrain, rightslicetrain, lefthalftest, rightslicetest, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(righthalftrain, leftslicetrain, righthalftest, leftslicetest)
	if param_name is not None and param is not None:
		a2 = Hemisphere(righthalftrain, leftslicetrain, righthalftest, leftslicetest, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'gestalt_slice_predict_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + 'gestalt_slice_predict_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		if save_name is None:
			save_array(mean_maps, 'gestalt_slice_predict_mean_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_gestalt_slice_predict_mean_maps')
	return mean_maps




if __name__ == '__main__':
	split_predict_slice_from_half_from_file('testimages_combined', slice_pix=30)

