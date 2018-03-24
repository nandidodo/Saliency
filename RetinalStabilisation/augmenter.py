

import numpy as np
import scipy
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

def translate(img, px, mode='constant'):
	assert type(px)==float or type(px)==int or len(px)==len(img.shape),'Pixels must either be a value or a tuple containing the number of pixels to move for each axis, and therefore musth ave the same dimension as the input image'
	return scipy.ndimage.interpolation.shift(img, px, mode=mode)


def augment_with_translations(img, num_augments=10,max_px_translate=4):
	augments = []
	#add the initial iamge
	augments.append(img)
	for i in range(num_augments):
		#get pixel values
		x_shift = int(np.random.uniform(low=(-1*max_px_translate), high=max_px_translate))
		y_shift = int(np.random.uniform(low=(-1*max_px_translate), high=max_px_translate))
		augment = translate(img, (x_shift, y_shift))
		augments.append(augment)

	#turn into numpy array and return
	augments = np.array(augments)
	return augments

def augment_with_copy(img, num_augments=10, copy=False):
	augments = []
	augments.append(img)
	for i in range(num_augments):
		#just copy
		#not sure if entirely new image is needed  i.e. np.copy
		if copy:
			augment = np.copy(img)
			augments.append(augment)
		if not copy:
			augments.append(img)

	augments = np.array(augments)
	return augments


def augment_dataset(dataset, num_augments, base_save_path=None, px_translate=4):
	#try to load dataset if it is a string
	if type(dataset) is str:
		dataset = np.load(dataset)
	#else assume it's okay
	assert num_augments>=0, 'Augments number cannot be negative. If zero, why are you doing this?'
	assert px_translate>1, 'Pixels to translate must be greater than zero'
	assert type(px_translate) is int, 'Pixels to translate must be integer'
	if base_save_path is not None:
		assert type(base_save_path) is str, 'Base save path must be a string'
	
	assert type(dataset) is np.ndarray, 'Dataset must be in the form of a numpy array, or a string which is the filename of a numpy array'

	#setup our base 
	augments = []
	augments = np.array(augments)
	copies = []
	copies = np.array(copies)

	#iterate over dataset
	for i in xrange(len(dataset)):
		augment = augment_with_translations(dataset[i], num_augments, px_translate)
		copy = augment_with_copy(dataset[i], num_augments)
		augments = np.vstack(augments, augment)
		copies = np.vstack(copies, copy)

	#just to make sure
	augments = np.array(augments)
	copies = np.array(copies)

	#if save
	if base_save_path is not None:
		np.save(base_save_path + "_augments", augments)
		np.save(base_save_path + "_copies", copies)

	return augments, copies


#first I need to check if this actally works, which I will do here
if __name__ == '__main__':
	#import mnist
	(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
	print xtrain.shape
	print ytrain.shape
	print xtest.shape
	print ytest.shape
	#first test augment
	#augments = augment_with_translations(xtrain[0])
	#print type(augments)
	#print augments.shape
	#for i in xrange(len(augments)):
	#	plt.imshow(augments[i], cmap='gray')
	#	plt.show()

	#perfect!
	#now let's test copies
	#copies = augment_with_copy(xtrain[0])
	#print type(copies)
	#print copies.shape
	#for i in xrange(len(copies)):
	#	plt.imshow(copies[i], cmap='gray')
	#	plt.show()
	#it works also

	#now let's test entire dataset
	augments, copies = augment_dataset(xtest, num_augments = 5)
	print type(augments)
	print type(copies)
	print augments.shape
	print copies.shape
	for i in xrange(50):
		plt.imshow(augments[i])
		plt.show()

