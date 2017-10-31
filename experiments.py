# okay, this is where I actually run my experiments, as another kind of master file/script

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


seed = 8
np.random.seed(seed)

def normalise(data):
	return data.astype('float32')/255.0


def load_colour_split_cifar(test_up_to = None):
	# let's try something

	(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

	xtrain = normalise(xtrain)
	xtest = normalise(xtest)

	redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
	redtest, greentest, bluetest = split_dataset_by_colour(xtest)


	redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
	greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
	bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
	redtest = np.reshape(redtest, (len(redtest), 32,32,1))
	greentest = np.reshape(greentest, (len(greentest), 32,32,1))
	bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

	if test_up_to is not None:
		redtrain = redtrain[0:test_up_to,:,:,:]
		greentrain = greentrain[0:test_up_to,:,:,:]
		bluetrain = bluetrain[0:test_up_to,:,:,:]
		redtest = redtest[test_up_to:test_up_to*2,:,:,:]
		bluetest = bluetest[test_up_to:test_up_to*2,:,:,:]
		greentest = greentest[test_up_to:test_up_to*2,:,:,:]

	return redtrain, greentrain, bluetrain, redtest, greentest, bluetest


def load_half_split_cifar(col = 1, test_up_to = None):
	(xtrain, ytrain), (xtest,ytest) = cifar10.load_data()
	xtrain = normalise(xtrain)
	xtest = normalise(xtest)
	
	redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
	redtest, greentest, bluetest = split_dataset_by_colour(xtest)

	redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
	greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
	bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
	redtest = np.reshape(redtest, (len(redtest), 32,32,1))
	greentest = np.reshape(greentest, (len(greentest), 32,32,1))
	bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

	if test_up_to is not None:
		redtrain = redtrain[0:test_up_to,:,:,:]
		greentrain = greentrain[0:test_up_to,:,:,:]
		bluetrain = bluetrain[0:test_up_to,:,:,:]
		redtest = redtest[test_up_to:test_up_to*2,:,:,:]
		bluetest = bluetest[test_up_to:test_up_to*2,:,:,:]
		greentest = greentest[test_up_to:test_up_to*2,:,:,:]

	half1train, half2train = split_image_dataset_into_halves(redtrain)
	half1test, half2test = split_image_dataset_into_halves(redtest)
	
	return half1train, half2train, half1test, half2test

def load_spatial_frequency_split_cifar(test_up_to=None):
	(xtrain, ytrain), (xtest,ytest) = cifar10.load_data()
	xtrain = normalise(xtrain)
	xtest = normalise(xtest)
	
	redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
	redtest, greentest, bluetest = split_dataset_by_colour(xtest)

	redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
	greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
	bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
	redtest = np.reshape(redtest, (len(redtest), 32,32,1))
	greentest = np.reshape(greentest, (len(greentest), 32,32,1))
	bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

	if test_up_to is not None:
		redtrain = redtrain[0:test_up_to,:,:,:]
		greentrain = greentrain[0:test_up_to,:,:,:]
		bluetrain = bluetrain[0:test_up_to,:,:,:]
		redtest = redtest[test_up_to:test_up_to*2,:,:,:]
		bluetest = bluetest[test_up_to:test_up_to*2,:,:,:]
		greentest = greentest[test_up_to:test_up_to*2,:,:,:]

	lptrain = filter_dataset(redtrain, lowpass_filter)
	lptest = filter_dataset(redtest, lowpass_filter)
	hptrain = filter_dataset(redtrain, highpass_filter)
	hptest = filter_dataset(redtest, highpass_filter)

	#bptrain = filter_dataset(redtrain, bandpass_filter)
	#bptest = filter_dataset(redtest, bandpass_filter)

	return lptrain, lptest, hptrain, hptest
		
	
	


# okay, that sorts out our data, now let's get the model working

def run_colour_experiments(epochs = 1, save=True, test_up_to=None):

	redtrain, greentrain, bluetrain, redtest, greentest, bluetest = load_colour_split_cifar(test_up_to=test_up_to)

	#for really fast training, for debugging
	#redtrain = redtrain[0:10,:,:,:]
	#greentrain = greentrain[0:10,:,:,:]

	print "REDTRAIN"
	print redtrain.shape

	a1 = Hemisphere(redtrain, greentrain, redtest, greentest)
	a2 = Hemisphere(greentrain, redtrain, greentest, redtest)

	a1.train(epochs=epochs)
	a2.train(epochs=epochs)

	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	#saving functionality
	if save:
		save(errmaps, 'colour_red_green_errormaps')

	return errmaps

def run_half_split_experiments(epochs = 1, save=True,test_up_to=None):
	
	half1train, half2train, half1test, half2test = load_half_split_cifar(test_up_to=test_up_to)

	a1 = Hemisphere(half1train, half2train, half1test, half2test)
	a2 = Hemisphere(half2train, half1train, half2test, half1test)

	a1.train(epochs=10)
	a2.train(epochs=10)

	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	#saving functionality
	if save:
		save(errmaps, 'colour_red_green_errormaps')

	return errmaps

def run_spatial_frequency_split_experiments(epochs=1, save=True, test_up_to=None):
	
	lptrain, lptest, hptrain, hptest = load_spatial_frequency_split_cifar(test_up_to=test_up_to)

	a1 = Hemisphere(lptrain, hptrain, lptest, hptest)
	a2 = Hemisphere(hptrain, lptrain, hptest, lptest)

	a1.train(epochs=10)
	a2.train(epochs=10)

	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	#saving functionality
	if save:
		save(errmaps, 'colour_red_green_errormaps')

	return errmaps


def run_benchmark_image_set_experiments(epochs=100, save=True, test_up_to=None):
	imgs = load('BenchmarkIMAGES_images')
	print imgs.shape
	red, green,blue = split_dataset_by_colour(imgs)
	print red.shape
	redtrain, redtest = split_into_test_train(red)
	print redtrain.shape
	greentrain, greentest = split_into_test_train(green)
	print redtrain.shape
	print redtest.shape

	#compare images here
	for i in xrange(10):
		compare_two_images(redtrain[i], greentrain[i], reshape=True)

	a1 = Hemisphere(redtrain, redtrain, redtest, redtest)
	print "hemisphere initialised"
	
	a2 = Hemisphere(greentrain, greentrain, greentest, greentest)
	print "second hemisphere initialised"
	

	a1.train(epochs=epochs)
	print "a1 trained"
	
	a2.train(epochs=epochs)
	print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()
	
	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		save_array(mean_maps, 'benchmark_red_green_error_maps')
	return mean_maps
	

if __name__ == '__main__':
	#run_colour_experiments(epochs=1, save=False)
	#run_spatial_frequency_split_experiments(epochs=1, save=False)
	#run_half_split_experiments(epochs=1, save=False)
	#okay, for whatever reason this thing just massively overloads my computer, nd I don't know why, so let' sbreak it down to be honest
# I think the problem is that when we'er splitting it everything remains in memory, so it's completely crazy tbh, we can fix that, but it will be annoying af
	#run_benchmark_image_set_experiments(100)
	run_colour_split_experiments(100, save=False, test_up_to=500)

