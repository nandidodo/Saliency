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
from gestalt_models import *
from autoencoder_gestalt import *


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

	#compare images here
	#for i in xrange(10):
	#	compare_two_images(redtrain[i], greentrain[i], reshape=True)

	

	a1 = Hemisphere(redtrain, greentrain, redtest, greentest,verbose=True)
	a2 = Hemisphere(greentrain, redtrain, greentest, redtest)

	
	a1.train(epochs=epochs, get_weights=True)
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
	

def run_half_split_experiments(epochs = 1, save=True,test_up_to=None, history=True):
	
	half1train, half2train, half1test, half2test = load_half_split_cifar(test_up_to=test_up_to)

	a1 = Hemisphere(half1train, half2train, half1test, half2test)
	a2 = Hemisphere(half2train, half1train, half2test, half1test)

	his1 =a1.train(epochs=epochs)
	his2 = a2.train(epochs=epochs)
	
	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	#saving functionality
	if save:
		save(errmaps, 'colour_red_green_errormaps_split_half')

	if history:
		his1 = serialize_class_object(his1)
		his2 = serialize_class_object(his2)
		return (errmaps, his1, his2)

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
	imgs = load('BenchmarkDATA/BenchmarkIMAGES_images')
	imgs= normalise(imgs)
	print imgs.shape
	red, green,blue = split_dataset_by_colour(imgs)
	print red.shape
	redtrain, redtest = split_into_test_train(red)
	print redtrain.shape
	greentrain, greentest = split_into_test_train(green)
	print redtrain.shape
	print redtest.shape

	#compare images here
	#for i in xrange(10):
	#	compare_two_images(redtrain[i], greentrain[i], reshape=True)

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

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		save_array(mean_maps, 'benchmark_red_green_error_maps')
	return mean_maps


def run_spatial_frequency_split_experiments_images_from_file(fname, epochs=100, save=True, test_up_to=None, preview=False, verbose=False, param_name=None, param=None, save_name=None, test_all=False):
	imgs = load_array(fname)
	lp,hp,bp = imgs
	print lp.shape
	lp = normalise(lp)
	hp = normalise(hp)
	bp = normalise(bp)
	lptrain, lptest=  split_into_test_train(lp)
	hptrain, hptest = split_into_test_train(hp)

	if preview:
		for i in xrange(10):
			compare_two_images(lptrain[i], hptrain[i], reshape=True)

	if param_name is None or param is None:
		a1 = Hemisphere(lptrain, hptrain, lptest, hptest)
	if param_name is not None and param is not None:
		a1 = Hemisphere(lptrain, hptrain, lptest, hptest, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(lptrain, hptrain, lptest, hptest)
	if param_name is not None and param is not None:
		a2 = Hemisphere(lptrain, hptrain, lptest, hptest, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	if test_all:
		a1=Hemisphere(lp, hp, lp, hp)
		a2=Hemisphere(hp,lp,hp,lp)
	

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
			save_array(mean_maps, 'benchmark_spfreq_red_green_error_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return mean_maps


def run_colour_split_experiments_images_from_file_with_all_hyperparams(fname,epochs=100, save=False, test_up_to=None, verbose = False,lrate=0.001,decay=1e-6, momentum=0.9, nesterov=True, shuffle=True, loss='binary_crossentropy',activation='relu', dropout=0.3, padding='same', save_name = None, batch_size=25, optimizer=None):
	imgs = load(fname)
	imgs= normalise(imgs)
	red, green,blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_into_test_train(red)
	greentrain, greentest = split_into_test_train(green)

	#setup default optimizer
	if optimizer is None:
		optimizer = optimizers.SGD(lr =lrate, decay=decay, momentum = momentum, nesterov = nesterov)

	a1=Hemisphere(redtrain, greentrain, redtest, greentest, batch_size=batch_size, dropout=dropout,activation=activation, padding=padding, optimizer =optimizer, epochs=epochs, loss=loss)
	a2=Hemisphere(greentrain, redtrain, greentest, redtest, batch_size=batch_size, dropout=dropout,activation=activation, padding=padding, optimizer =optimizer, epochs=epochs, loss=loss)
	

	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"
	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	mean_maps = mean_map(errmap1, errmap2)

	if save:
		if save_name is None:
			save_array(mean_maps, 'benchmark_red_green_error_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return redtest, preds1, errmap1, mean_maps



def run_colour_split_experiments_images_from_file(fname,epochs=100, save=True, test_up_to=None, preview = False, verbose = False, param_name= None, param = None, save_name = None, test_all=False):
	imgs = load(fname)
	imgs= normalise(imgs)
	red, green,blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_into_test_train(red)
	greentrain, greentest = split_into_test_train(green)

	if preview:
		for i in xrange(10):
			compare_two_images(redtrain[i], greentrain[i], reshape=True)

	if param_name is None or param is None:
		a1 = Hemisphere(redtrain, greentrain, redtest, greentest)
	if param_name is not None and param is not None:
		a1 = Hemisphere(redtrain, greentrain, redtest, greentest, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(greentrain, redtrain, greentest, redtest)
	if param_name is not None and param is not None:
		a2 = Hemisphere(greentrain, redtrain, greentest, redtest, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	if test_all:
		a1=Hemisphere(red, green, red, green)
		a2=Hemisphere(green,red,green,red)
	

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
			save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		if save_name is None:
			save_array(mean_maps, 'benchmark_red_green_error_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return mean_maps



def run_all_colour_split_experiments_images_from_file(fname,epochs=100, save=True, test_up_to=None, preview = False, verbose = False, param_name= None, param = None, save_name = None, test_all=False):
	imgs = load(fname)
	imgs= normalise(imgs)
	red, green,blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_into_test_train(red)
	greentrain, greentest = split_into_test_train(green)
	bluetrain, bluetest = split_into_test_train(blue)

	a1 = Hemisphere(redtrain, greentrain, redtest, greentest)
	a2 = Hemisphere(greentrain, redtrain, greentest, redtest)
	a3 = Hemisphere(redtrain, bluetrain, redtest, bluetest)
	a4 = Hemisphere(bluetrain, redtrain, bluetest, redtest)
	a5 = Hemisphere(greentrain, bluetrain, greentest, bluetest)
	a6 = Hemisphere(bluetrain, greentrain, bluetest, greentest)
	
	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a3.train(epochs=epochs)
	if verbose:
		print "a3 trained"

	a4.train(epochs=epochs)
	if verbose:
		print "a4 trained"
	
	a5.train(epochs=epochs)
	if verbose:
		print "a25trained"

	a6.train(epochs=epochs)
	if verbose:
		print "a6 trained"

	a1.plot_results()
	a2.plot_results()
	a3.plot_results()
	a4.plot_results()
	a5.plot_results()
	a6.plot_results()


	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)
	preds3, errmap3 = a3.get_error_maps(return_preds = True)
	preds4, errmap4 = a4.get_error_maps(return_preds=True)
	preds5, errmap5 = a5.get_error_maps(return_preds = True)
	preds6, errmap6 = a6.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')

	#if verbose:
	#	print errmap1[0]
	
	#a1.plot_error_maps(errmap1, predictions=preds1)
	#a2.plot_error_maps(errmap2,predictions=preds2)
	
	err_maps=(errmap1, errmap2, errmap3, errmap4, errmap5, errmap6)
	mean_map = mean_maps(err_maps)
	a1.plot_error_maps(mean_map)

	if save:
		if save_name is None:
			save_array(mean_map, 'benchmark_red_green_error_maps_all')
		if save_name is not None:
			save_array(mean_map, save_name + '_mean_maps')
	return mean_map


def compare_error_map_to_salience_map(err_fname, sal_fname, start = 100, gauss=False):
	tup = load(err_fname)
	errmap = tup[2]
	print "ERRMAP:"
	print errmap.shape
	salmap = load(sal_fname)
	N = int(len(salmap)/10)
	salmap=salmap[1710:1900,:,:,0]
	shape = salmap.shape
	salmap = np.reshape(salmap, (shape[0], shape[1], shape[2])) 
	print "SALMAP:"
	print salmap.shape
	preds = tup[1]
	test = tup[0]
	preds = np.reshape(preds, (shape[0], shape[1], shape[2]))
	test = np.reshape(test,(shape[0], shape[1], shape[2]))
	
	for i in xrange(50):
		if not gauss:
			imgs = (test[start + i], preds[start + i], errmap[start + i], salmap[start + i])
			titles=('test image', 'prediction', 'error map', 'target salience map')
			compare_images(imgs, titles)
		if gauss:
			sigma=2
			errm = gaussian_filter(errmap[start+i],sigma)
			imgs = (test[start + i], preds[start + i],errm, salmap[start + i])
			titles=('test image', 'prediction', 'error map', 'target salience map')
			compare_images(imgs, titles)
	#compare_saliences(errmap, salmap)

def compare_mean_map_to_salience_map(mmap_fname, sal_fname, start = 100, gauss=False, N = 50):
	mmap = load(mmap_fname)
	print "MEAN MAP:"
	print mmap.shape
	salmap = load(sal_fname)
	salmap=salmap[1710:1900,:,:,0]
	shape = salmap.shape
	salmap = np.reshape(salmap, (shape[0], shape[1], shape[2])) 
	print "SALMAP:"
	print salmap.shape
	for i in xrange(N):
		if not gauss:
			compare_two_images(mmap[start + i], salmap[start+i], 'mean error map', 'target salience map')
		if gauss:
			sigma = 2
			errm = gaussian_filter(mmap[start+i])
			compare_two_images(errm, salmap[start+i], 'mean error map', 'target salience map')
	



	
def hyperparam_grid_search(param_name, param_list, input_fname, save_base, epochs=100, error=True, error_list = False, sal_map_fname = 'testsaliences_combined', fn=run_colour_split_experiments_images_from_file):
	N = len(param_list)
	for i in xrange(N):
		save_name = save_base + '_' + param_name + '_test_'+str(i)
		mean_maps = fn(input_fname, epochs=epochs, param_name = param_name, param = param_list[i],save_name = save_name)
		if error:
			#we load and process the sal maps
			salmaps = load(sal_map_fname)
			salmaps = salmaps[:,:,:,0]
			shape = salmaps.shape
			salmaps = np.reshape(salmaps,(shape[0], shape[1], shape[2]))
			if error_list:
				err, errlist = get_errors(mean_maps, salmaps, error, error_list, save_name='save_base' + '_' + str(param) + '_'+str(param_list[i]) + '_errors')
			if not error_list:
				err = get_errors(mean_maps, salmaps, error, error_list, save_name='save_base' + '_' + str(param) + '_'+str(param_list[i]) + '_errors')


def multi_hyperparam_grid_search(param_names, param_lists, input_fname, save_bases, epochs=100, error=True, error_list = False, sal_map_fname = 'testsaliences_combined', fn=run_colour_split_experiments_images_from_file):
	N = len(param_names)
	assert N == len(param_lists) == len(save_bases), "each hyperparam must have param list and save base"
	for i in xrange(N):
		hyperparam_grid_search(param_names[i], param_lists[i], input_fname, save_bases[i], epochs=epochs, error=error, error_list=error_list, sal_map_fname = sal_map_fname, fn=fn)


#we need a way to get the error and accuracies or whatever, but we'll have to add that in a bit, so I don't know!


def try_gestalt_model_for_normal(fname,epochs=100, both=True):
	imgs = load_array(fname)
	imgs = imgs.astype('float32')/255.
	red, green, blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_first_test_train(red)
	greentrain, greentest = split_first_test_train(green)
	shape = redtrain.shape

	model = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
	model.compile(optimizer='sgd',loss='mse')
	callbacks = build_callbacks("./")
	his=model.fit(redtrain, greentrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(redtest, greentest), callbacks=callbacks)

	if both:
		model2 = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
		model2.compile(optimizer='sgd', loss='mse')
		his2 = model2.fit(greentrain, redtrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(greentest, redtest), callbacks=callbacks)

	print "MODEL FITTED"

	preds1 = model.predict(redtest)
	history = serialize_class_object(his)
	res = [history, preds, redtest, greentest]
	save_array(res, "STANDARD_WITH_GESTALT_AUTOENCODER_MODEL_1")

	if both:
		preds2 = model2.predict(greentest)
		history2 = serialize_class_object(his2)
		res2 = [history2, preds2, greentest, redtest]
		save_array(res2, "STANDARD_WITH_GESTALT_AUTOENCODER_MODEL_2")
	



if __name__ == '__main__':
	#run_colour_experiments(epochs=1, save=False)
	#run_spatial_frequency_split_experiments(epochs=1, save=False)
	#run_half_split_experiments(epochs=1, save=False)
	#okay, for whatever reason this thing just massively overloads my computer, nd I don't know why, so let' sbreak it down to be honest
# I think the problem is that when we'er splitting it everything remains in memory, so it's completely crazy tbh, we can fix that, but it will be annoying af
	#run_benchmark_image_set_experiments(20)
	#run_colour_experiments(5, save=False, test_up_to=10)
	#run_colour_split_experiments_images_from_file('testimages_combined', epochs=50)
	#compare_error_map_to_salience_map('BenchmarkIMAGES_images', 'BenchmarkIMAGES_output')
	#compare_error_map_to_salience_map('testimages_combined_imgs_preds_errmaps', 'testsaliences_combined', gauss=True)

	#compare_mean_map_to_salience_map('benchmark_red_green_error_maps', 'testsaliences_combined', gauss=True)

	#run_colour_split_experiments_images_from_file('testimages_combined', epochs=50, test_all=True, save_name="all_errmaps")
	#run_spatial_frequency_split_experiments_images_from_file('benchmark_images_spatial_frequency_split', epochs=50,test_all=True, save_name='spfreq_errmaps')
	#run_half_split_experiments()
	#run_all_colour_split_experiments_images_from_file('testimages_combined', epochs=50, test_all=True, save_name="all_errmaps_all_colour_combinations")


	#quick hyperparam test before the main event
	"""
	param_names=("lrate","momentum")
	lrates=(0.001, 0.002,0.01,0.1,0.0001,0.005,0.05)
	moms = (0.9, 0.5,0.8,0.99,0.3,0)
	res = []
	for lrate in lrates:
		result = run_colour_split_experiments_images_from_file_with_all_hyperparams('testimages_combined', epochs=50,lrate=lrate)
		res.append(result)
	save_array(res, "hyperparam_test_lrates") 
	res2=[]
	for mom in moms:
		result = run_colour_split_experiments_images_from_file_with_all_hyperparams('testimages_combined', epochs=50, momentum=mom)
		res2.append(result)
	save_array(res2, "hyperparam_test_momentums")
	"""
	#lrates=(0.001, 0.002,0.01,0.1,0.0001,0.005,0.05)
	#process_hyperparams_error("hyperparam_test_lrates", "testsaliences_combined", "lrate", lrates)

	#okay, here are the actual history tests
	#maps, his1, his2 = run_half_split_experiments(epochs=1, save=False)
	#print type(his1)
	#print his1
	#save_array((his1, his2), "history_experiment")

	# okay, this works. what shuold be my next steps in modelling this?
	# so we can try to get the gestalts working and play aroudn with those numbers
	#that seems like the most reasonable solution to be honest, so let's try to get that working
	#and then see what we can obtain from that. if wecan do thattonight, and get a reasonable looking model, then I wuold be very happy indeed. so let's try that!


	#okay, here's me trying my probably better and more successful gestalt models
	#for the standard autoencoding task
	try_gestalt_model_for_normal("testsaliences_combined", epochs=1, both=True)
	
		




















# the next step, depending on what we say to richard, is the hyperparam search. that isn't that difficult, but that's the default step. we can write something to dothat, and to be hoenst I should probably just do it now, as I'm not realistically going to be able to do any significantly useful other stuff in like the half an hour I've got now. I'll upload my react/vue stuff to github, and then see where i stand, I think. Getting hyperparam search working andthe save files set up, so Ican just run it tonight while doing stuff with mycah would be the ideal, so I shuold look at it and see how it's going and where we get up to... so let's do that!

# one very simpel thing we we don't actually compare the mean maps to the saliecne traces. maybe if we did that would help. that's pretty trivial to do all things considered, so we should look at making that work!

# first we're going to run this and then test it by hand. we're going to have to have some eval scripts to evaluate differences for us, and then test different smoothing methods, such as gaussian smoothing or whatever to see if there's anything useful there. furthermore, and we are kind of doomed here, we'll need to figure out just hwt exactly we're doing re hyperparameters and stuff,  but a lot of that is incremental and boring. We could also start writign stuff up, but basiclaly just talk to richard. that seems to be the way. okay, now I'ev got stuff going I think I'm going to go to my desk and start working through the react-yelp tutorial to see if there's anything cool or useful there... should be fun!


# okay, so what are our plans for today wrt the phd stuff? This is actually going okay, because the saliences do often seem to track the error differences, which is cool
# so, what are we going to do? well, there are two stories we can tell, one is just about the hemispheres generating edge detection kind of automatically, whic his really awesome, we should definitely show richard thatand tie that into predictive processing and biological models, as it's really cool. could be a minimum publishable unit (as well as the autism!?). second we would try to recreate the salienec patterns by testing them with our thing. that's really what we needto get done now, I feel so let's get on that
# the other trouble is that the current collect dirs doesn't actually do stuff in order either, which is unfortuante, I think so we'll need to rewrite that one to do so also, and then we just need to train the model and compare which is fair enoguh. at some later point we'll need to start wriging plotting functions so we can compare training performance over time, and start seriosuly consdiering optimisations and scripts for hyperparameter search. moreoever, we'll need to start trying the different ones. that's where I'll just need to go into the office and get some serious computatoinal power
# but yeah, I think today the aim is just to find this and get it running while we actually work on the other important stuff, which is greater importance re Enyo, such as figuring out how to do a react application. that seems very important, and must be done

# right, that means I actually need to start working at stuff. so let's do that?
# okay, we emailed richard, next step is generating the function that combines the data into a big array, andthen training with it, so let's do that


# so basically we have found that the things it has trouble predicting, and thus the most "informative" parts of the picture are basically the edges, nwo this is actually quite interesting, and if I'mfeeling pretty dodgy, I could write a paper on this lol, as it would be quite interesting, and could see why we do edges, and then we would do a fucking thing where we see edges and perhaps try to integrate it with gestalts, and so forth, and that could be really really interesting, but Ithink it's quite unlikely to be honest but at least that's something cool we've found. one thing we need to do is to improve the plot error map functoin to show all three, so let's do that, and then lets work on other stuff - i.e. we'll perhaps, I do't know what the next step is - doing gaussian smoothing seems to be important, also training on the huge image corpus and then cross validating, with the actual responses. so let's work on that and get that done today, then we can do some smiple html and css which could be fun. tomorrow we'll prepare what we're actually going to say to richard!


# we should actually talk to richard about that... what are our next steps, however? I'm not actually sure of any of this, so I should have a look to see what we've got
# we could perhaps try to write a paper about it - could be interesting. next steps are trying it out, trying to actually properly cmopare the images to the things. let's have a look at that now

# I think I know what the issue is, actually, which is interesting. I'm pretty sure it relates to the fact that python dictionaries aren't ordered, and when we do our file system, it parses a dictoinary, which is going to be really really irritating. We're going to have to have a better unciton
# OTOH the salience maps often actually look kind of like the outlines which we see with the reconstructions, so that's really promising. we should try to solev these issues, I feel

# yeah, there aretwo problems. firstly, it apparently completely ignores each file, secodnly it actually doesn't read them in in the right order AT ALL, so I'm really not sure how I should do this to be perfectly honest. I really don't know. the trouble is they come in directories, so I'm not sure how to maintain the order. let's go on stackoverflow to have a look!
# okay, there are only 100, it just orders them in even numbers for whatever dumb reason. okay, we can solve this, let's get to it!
