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

if len(sys.argv) >1:
	run_num = sys.argv[1]

def split_dataset_half_small_section(dataset,split_width):
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
	


def plot_slice_splits(leftsplit, rightsplit, leftslice, rightslice, show=True):	
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
	leftsplits, rightsplits, leftslices, rightslices = split_dataset_half_small_section(dataset, split_width=split_width, history=True)
	N = len(dataset)
	for i in xrange(N):
		plot_slice_splits(leftsplits[i], rightsplits[i], leftslices[i], rightslices[i])
	


def split_half_image_experiments_from_file(fname, epochs=100, save=True, test_up_to=None, preview=True, verbose=False, param_name=None, param=None, save_name=None, test_all=False):
	#we'll do this in the three dimensional test
	imgs = load_array(fname)
	train, test = split_into_test_train(imgs)
	#train = two_dimensionalise(train, reshape=False, expand=True)
	#test = two_dimensionalise(test, reshape=False, expand=True)
	train, greentrain, bluetrain = split_dataset_by_colour(train)
	test, greentest, bluetest = split_dataset_by_colour(test)
	halftrain1, halftrain2 = split_image_dataset_into_halves(train)
	halftest1, halftest2 = split_image_dataset_into_halves(test)
	
	if preview:
		for i in xrange(10):
			compare_two_images(halftrain1[i], halftrain2[i], reshape=True)

	if param_name is None or param is None:
		a1 = Hemisphere(halftrain1, halftrain2, halftrain1, halftrain2)
	if param_name is not None and param is not None:
		a1 = Hemisphere(halftrain1, halftrain2, halftrain1, halftrain2, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(halftrain2, halftrain1,halftest2, halftest1)
	if param_name is not None and param is not None:
		a2 = Hemisphere(halftrain2, halftrain1, halftest2, halftest1, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	if test_all:
		a1=Hemisphere(halftrain1, halftrain2, halftrain1, halftrain2)
		a2=Hemisphere(halftrain2,halftrain1,halftrain2,halftrain1)
	

	hist1 = a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	hist2 = a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),"gestalt_split_imgs_preds_errmaps")
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + "gestalt_split_imgs_preds_errmaps")

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
	if history:
		return (mean_maps, hist1, hist2)
	return mean_maps


def two_dimensionalise(arr, col=0, reshape=True, expand=False):
	shape = arr.shape
	assert len(shape) == 4, "input array must be four dimensional (3d img + num of imgs)"
	arr = arr[:,:,:,col]
	if reshape:
		arr =  np.reshape(arr, (shape[0], shape[1], shape[2]))
	if expand:
		arr = np.reshape(arr, (shape[0], shape[1], shape[2], 1))
	return arr


def split_predict_slice_from_half_from_file(fname,slice_pix=30, epochs=100, save=True, test_up_to=None, preview=False, verbose=False, param_name=None, param=None, save_name=None, test_all=False, his=True):
	#we'll do this in the three dimensional test
	imgs = load_array(fname)
	train, test = split_into_test_train(imgs)
	#train =np.reshape(train[:,:,:,0], (train.shape[0], train.shape[1], train.shape[2]))
	#test =np.reshape(test[:,:,:,0], (test.shape[0], test.shape[1], test.shape[2]))
	train = two_dimensionalise(train, reshape=False, expand=True)
	test = two_dimensionalise(test, reshape=False, expand=True)
	lefthalftrain, righthalftrain, leftslicetrain, rightslicetrain = split_dataset_half_small_section(train, slice_pix)
	lefthalftest, righthalftest, leftslicetest, rightslicetest = split_dataset_half_small_section(test, slice_pix)

	if param_name is None or param is None:
		a1 = Hemisphere(lefthalftrain, rightslicetrain, lefthalftest, rightslicetest, verbose=True)
	if param_name is not None and param is not None:
		a1 = Hemisphere(lefthalftrain, rightslicetrain, lefthalftest, rightslicetest, param_name=param, verbose=True)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(righthalftrain, leftslicetrain, righthalftest, leftslicetest)
	if param_name is not None and param is not None:
		a2 = Hemisphere(righthalftrain, leftslicetrain, righthalftest, leftslicetest, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	his1 = a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	his2 = a2.train(epochs=epochs)
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

	if his:
		return (mean_maps, his1, his2)
	return mean_maps

#test



if __name__ == '__main__':
	#split_predict_slice_from_half_from_file('testimages_combined', slice_pix=30)
	#fname = "BenchmarkDATA/BenchmarkIMAGES_images"
	#dataset = load_array(fname)
	#show_split_dataset_with_slices(dataset,split_width=10)
	#mean_maps, his1, his2= split_predict_slice_from_half_from_file("testimages_combined", slice_pix=20,epochs=50) 
	#we save history here
	#save_array((his1, his2), "gestalt_history_callback_test")
	mean_maps, his1, his2  = split_half_image_experiments_from_file("testimages_combined", epochs=3, save=False)
	#save_array((his1, his2), "gestalt_history_callback_test")

