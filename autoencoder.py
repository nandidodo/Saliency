# okay, the aim of this is to refactor our autoencoder code into a class based model where it can be much more easily utilised and understood. that shouldn't be too difficult,, although I'm not quite sure the best practice to do this a.) in python, and b.) within the keras api, which is going to be unfortuante, of course, but also kind of fun and interesting!, so let's get to it

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

# define reasonable defaults
batch_size = 25
dropout = 0.3
verbose = False
architecture = None
activation = 'relu'
padding = 'same'

epochs = 20

lrate = 0.01
decay = 1e-6
momentum = 0.9
nesterov = True
shuffle = True

optimizer = optimizers.SGD(lrate =lrate, decay=decay, momentum = momentum, nesterov = nesterov)



class Hemisphere(object):
	
	def __init__(input_data, output_data,test_input, test_output, batch_size = batch_size, dropout = dropout, verbose = verbose, architecture = architecture, activation = activation, padding = padding, optimizer = optimizer, epochs = epochs):
		#init our paramaters
		self.input_data = input_data
		self.output_data = output_data
		self.test_input = test_input
		self.test_output = test_output
		self.batch_size = batch_size
		self.dropout = dropout
		self.verbose = verbose
		self.architecture = architecture
		self.activation = activation
		self.padding = padding
		self.optimizer = optimizer
		self.epochs = epochs

		#next we do some asserts
		self.shape = input_data.shape
		assert shape == output_data.shape, 'input and output data do not have the same shape'

		#next we define our model, we will normally do something with architecture here, but we're not going o do that yet, so we'll have a placeholder
		if architecture is not None:
			# initialies and set it up, but we've not impleneted it yet
			raise NotImplementedError
			pass
		if architecture is None:

			input_img = Input(shape=(self.shape[1], self.shape[2], self.shape[3])

			x = Conv2D(16, (3, 3), activation=self.activation, padding=self.padding)(input_img)
			if verbose:
				print x.shape
			#x = MaxPooling2D((2, 2), padding='same')(x)
			#print x.shape
			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			#x = MaxPooling2D((2, 2), padding='same')(x)
			#print x.shape
			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			encoded = MaxPooling2D((2, 2), padding='same')(x)
			if verbose:
				print encoded.shape
				print "  "

			# at this point the representation is (4, 4, 8) i.e. 128-dimensional

			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(encoded)
			if verbose:
				print x.shape
			x = UpSampling2D((2, 2))(x)
			if verbose:
				print x.shape
			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			#x = UpSampling2D((2, 2))(x)
			#print x.shape
			x = Conv2D(16, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			#x = UpSampling2D((2, 2))(x)
			#print x.shape
			decoded = Conv2D(1, (3, 3), activation='sigmoid', padding=self.padding)(x)
			if verbose:
				print decoded.shape

			#we then define our model
			self.model = Model(input_img, decoded)



	# okay, so we begin the functions here
	def train(self, epochs = self.epochs, optimizer = self.optimizer, shuffle=True, callbacks = None)
		print "Model training:"
		self.model.fit(self.input_data, self.output_data, epochs=epochs, optimizer = optimizer, shuffle = shuffle, callbacks = callbacks)
		print "Training complete

	def predict(self, test_data = None):
		if test_data is not None:
			return self.model.predict(test_data)
		if test_data is None:
			return self.model.predict(self.test_input)
	
	def get_error_maps(self, input_data = None, predictions= None):
		# we get all combinations of things here for behaviour
		# we also reshape them here, for the rest of all time, so my functoins are easy
		shape = (self.shape[0], self.shape[1], self.shape[2])
		if input_data is None && predictions is None:
			maps= np.absolute(self.test_input, self.test_output)
			return np.reshape(maps, shape)
		if input_data is None && predictions is not None:
			maps np.absolute(self.test_input - predictions)
			return np.reshape(maps, shape)
		if input_data is not None && predictions is None:
			maps np.absolute(input_data - self.test_output)
			return np.reshape(maps, shape)
		if input_data is not None && predictions is not None:
			maps np.absolute(input_data - predictions)
			return np.reshape(maps, shape)

	


	def plot_error_maps(self, error_maps = None, N = 10, original_images = None):
		if error_maps is None:
			error_maps = get_error_maps()
		if original_images is None:
			original_images = np.reshape(self.test_output, (self.shape[0], self.shape[1], self.shape[2]))
		for i in xrange(N):
			compare_two_images(original_images[i], error_maps[i], 'Original', 'Error Map')

	def generate_mean_maps(self, error_maps1, error_maps2, N = -1):
		n = len(error_maps1)
		assert len(error_maps2) != n, 'different numbers of maps in each'
		if N == -1:
			#so nothing specified we do the full lunch
			N = n
		mean_maps = []
		for i in xrange(N):
			mean_maps.append(mean_map(error_maps1[i], error_maps2[i]))
		mean_maps = np.array(mean_maps)
		return mean_maps

	def plot_mean_error_maps(self,mean_maps, N = 10):
		if N == -1:
			N = len(mean_maps)
		plot_error_maps(mean_maps, N)
		

		
