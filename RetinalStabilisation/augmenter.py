

import numpy as np
import scipy
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

def translate(img, px, mode='constant'):
	#print "in translate"
	#print px
	#print len(px)
	#print len(img.shape)
	#print img.shape
	assert type(px) is float or type(px) is int or len(px)==len(img.shape),'Pixels must either be a value or a tuple containing the number of pixels to move for each axis, and therefore musth ave the same dimension as the input image'
	return scipy.ndimage.interpolation.shift(img, px, mode=mode)


def augment_with_translations(img, num_augments=10,max_px_translate=4):
	assert len(img.shape)==2,'Image must be two dimensional'
	assert num_augments>=0, 'Number of augments must be positive'
	assert max_px_translate>=0, 'Pixels to translate must be positive'
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

def augment_with_translation_deterministic(img, num_augments, px_translate):
	
	assert len(img.shape)==2,'Image must be two dimensional'
	assert num_augments>=0, 'Number of augments must be positive'
	assert px_translate>=0, 'Pixels to translate must be positive'

	# it translates by the max amount, but whether x translation, ytranslation or both is random
	augments = []
	#don't have the initial image this time!
	#augments.append(img)
	#sh = img.shape
	#img = np.reshape(img, (sh[0], sh[1],sh[2]))
	for i in range(num_augments):
		shift_direction = 0
		rand = np.random.uniform(low=0, high=1)
		#print i
		if rand >=0.33 and rand <=0.66:
			shift_direction=1
		if rand >0.66 and rand <=1:
			shift_direction=2
		if shift_direction==0:
			#print px_translate
			#print img.shape
			augment = translate(img, (px_translate, 0))
			augments.append(augment)
		if shift_direction==1:
			#print px_translate
			#print img.shape
			augment = translate(img, (0, px_translate))
			augments.append(augment)
		if shift_direction==2:
			#print px_translate
			#print img.shape
			augment = translate(img, (px_translate, px_translate))
			augments.append(augment)

	augments = np.array(augments)
	#print "in augment deterministic"
	#print augments.shape
	return augments

def augment_with_copy(img, num_augments=10, copy=False):
	assert len(img.shape)==2, 'Image must be two dimensional'
	assert num_augments >=0, 'Number of augments must be positive'
	assert type(copy) is bool, 'Copy is a boolean flag'
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


def create_translation_invariance_test_datasets(dataset, num_augments, save_base,min_px_translate=0, max_px_translate=10, steps=5):
	print "In create invariance test datasets"
	if type(dataset) is str:
		dataset = np.load(dataset)

	assert num_augments>=0, 'Number of augments acnnot be zero or negative'
	assert min_px_translate >=0 and min_px_translate <=max_px_translate, 'Min pixels must be positive and less than max pixels (obviously)'
	assert max_px_translate >0, 'if 0 or less, why are you using this function'
	assert steps >0, 'steps must be greater than 0'
	assert type(save_base) is str and len(save_base)>0, 'Save base must be a valid string'

	step_size = (max_px_translate-min_px_translate)//steps
	#print "step size"
	#print step_size
	#print steps
	#now begin the loop to create the datasets
	for i in range(steps):
		px_translate = min_px_translate + (i * step_size)
		#base case
		augments = augment_with_translation_deterministic(dataset[0], num_augments, px_translate)
		for j in xrange(len(dataset)-1):
			augment = augment_with_translation_deterministic(dataset[j+1], num_augments, px_translate)
			#print "in loop augment shape"
			#print augment.shape
			#print step_size
			#print steps
			augments = np.concatenate((augments, augment))
			#print augments.shape
		#augments = np.array(augments)
		save_name = save_base + '_' + str(px_translate) + 'pixels_translate'
		print "final augments shape"
		print augments.shape
		np.save(save_name, augments)

	#and that's it. simple functoin, I think
	return

def generate_labels(labels, num_augments,save_name=None):

	if save_name is not None:
		assert type(save_name) is str and len(save_name)>0,'Save name must be a valid string'

		#base case
	generated_labels = np.full(num_augments+1, labels[0])
	#loop!
	for i in xrange(len(labels)-1):
		aug_labels = np.full(num_augments+1, labels[i+1])
		generated_labels = np.concatenate((generated_labels, aug_labels))
	generated_labels = np.array(generated_labels)
	print "In generated labels: ", generated_labels.shape

	if save_name is not None:
		np.save(save_name, generated_labels)

	return generated_labels

def augment_labels_func(label, num_augments):
	labels = np.full(num_augments+1, label)
	return labels


def augment_dataset_discriminative(dataset, labels, num_augments, base_save_path=None, px_translate=4):
	
	assert num_augments>=0,'Number of augments cannot be negative'
	if base_save_path is not None:
		assert type(base_save_path) is str and len(base_save_path)>0, 'Save path must be a valid string'
	assert px_translate>=0, 'Pixels translate cannot be negative'

	if type(dataset) is str:
		dataset = np.load(dataset)
	if type(labels) is str:
		labels = np.load(labels)

	assert len(dataset) == len(labels), 'must have same number of labels as data items'

	#setup base case
	augments = augment_with_translations(dataset[0], num_augments, px_translate)
	copies = augment_with_copy(dataset[0], num_augments)
	aug_labels = augment_labels_func(labels[0], num_augments)
	copy_labels = augment_labels_func(labels[0], num_augments)

	for i in xrange(len(dataset)-1):
		augment= augment_with_translations(dataset[i+1], num_augments, px_translate)
		copy = augment_with_copy(dataset[i+1], num_augments)
		new_labels = augment_labels_func(labels[i+1], num_augments)
		augments = np.concatenate((augments, augment))
		print augments.shape
		copies = np.concatenate((copies, copy))
		print copies.shape
		aug_labels = np.concatenate((aug_labels, new_labels))
		print aug_labels.shape
		copy_labels = np.concatenate((copy_labels, new_labels))

	augments = np.array(augments)
	copies = np.array(copies)
	augment_labels = np.array(aug_labels)
	copy_labels = np.array(copy_labels)

	if base_save_path is not None:
		np.save(base_save_path+'_aug_data', augments)
		np.save(base_save_path+'_aug_labels', aug_labels)
		np.save(base_save_path+'_copy_data', copies)
		np.save(base_save_path+'_copy_labels', copy_labels)

	return augments, copies, aug_labels, copy_labels


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

	#setup base case
	augments = augment_with_translations(dataset[0], num_augments, px_translate)
	copies = augment_with_copy(dataset[0], num_augments)

	#iterate over dataset
	for i in xrange(len(dataset)-1):
		augment = augment_with_translations(dataset[i+1], num_augments, px_translate)
		copy = augment_with_copy(dataset[i+1], num_augments)
		#this does not work!
		#augments.append(augment)
		#copies.append(copy)
		#print augment.shape
		#print augments.shape
		augments = np.concatenate((augments, augment))
		copies = np.concatenate((copies, copy))
		print "in augment loop"
		print augments.shape
		print copies.shape

	#just to make sure
	augments = np.array(augments)
	copies = np.array(copies)

	#if save
	if base_save_path is not None:
		np.save(base_save_path + "_augments", augments)
		np.save(base_save_path + "_copies", copies)

	return augments, copies

def test_discriminative_data():
	test_aug = np.load('data/discriminative_test_aug_data.npy')
	test_copy = np.load('data/discriminative_test_copy_data.npy')
	print "test aug data: ", test_aug.shape
	print "test copy data: " , test_copy.shape

	test_aug_labels = np.load('data/discriminative_test_aug_labels.npy')
	test_copy_labels = np.load('data/discriminative_test_copy_labels.npy')
	print "test aug labels: ", test_aug_labels.shape
	print "test_copy_labels: " , test_copy_labels.shape

	train_aug = np.load('data/discriminative_train_aug_data.npy')
	train_copy = np.load('data/discriminative_train_copy_data.npy')
	print "train aug data: " , train_aug.shape
	print "train copy data: " , train_copy.shape

	train_aug_labels = np.load('data/discriminative_train_aug_labels.npy')
	train_copy_labels = np.load('data/discriminative_train_copy_labels.npy')
	print "train aug labels: " , train_aug_labels.shape
	print "train copy labels: " , train_copy_labels.shape



#first I need to check if this actally works, which I will do here
if __name__ == '__main__':
	#import mnist
	(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
	#print xtrain.shape
	#print ytrain.shape
	#print xtest.shape
	#print ytest.shape
	##first test augment
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
	#augments, copies = augment_dataset(xtest, num_augments = 5)
	#print type(augments)
	#print type(copies)
	#print augments.shape
	#print copies.shape
	#for i in xrange(50):
	#	plt.imshow(augments[i])
	#	plt.show()
	#okay, augmenter works, that's wonderful

	#now actually create the dataset
	#save_path= "data/mnist_dataset"
	#create train datasets
	#augment_dataset(xtrain, num_augments=10,base_save_path = save_path+"_train")
	#create test dataset
	#augment_dataset(xtest, num_augments=10, base_save_path = save_path+"_test")

	#save_path = "data/mnist_invariance"
	#create_translation_invariance_test_datasets(xtest, 10, save_path )

	#save_path = "data/discriminative_train"
	#num_augments = 10
	#augment_dataset_discriminative(xtrain, ytrain,num_augments, save_path)
	#save_path="data/discriminative_test"
	#augment_dataset_discriminative(xtest, ytest, num_augments, save_path)
	#test_discriminative_data()
	#all is looking well here! yay!
