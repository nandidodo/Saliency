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
import keras

# define reasonable defaults
batch_size = 25
dropout = 0.3
verbose = False
architecture = None
activation = 'relu'
padding = 'same'

epochs = 20

lrate = 0.001
decay = 1e-6
momentum = 0.9
nesterov = True
shuffle = True
loss = 'binary_crossentropy'

optimizer = optimizers.SGD(lr =lrate, decay=decay, momentum = momentum, nesterov = nesterov)

default_callbacks = [keras.callbacks.TerminateOnNaN]


class Hemisphere(object):
	
	def __init__(self,input_data, output_data,test_input, test_output, batch_size = batch_size, dropout = dropout, verbose = verbose, architecture = architecture, activation = activation, padding = padding, optimizer = optimizer, epochs = epochs, loss=loss):

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
		self.loss = loss

		#next we do some asserts
		self.shape = input_data.shape
		assert self.shape == output_data.shape, 'input and output data do not have the same shape'

		#next we define our model, we will normally do something with architecture here, but we're not going o do that yet, so we'll have a placeholder
		if architecture is not None:
			# initialies and set it up, but we've not impleneted it yet
			raise NotImplementedError
			pass
		if architecture is None:

			input_img = Input(shape=(self.shape[1], self.shape[2], self.shape[3]))

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
			self.model.compile(optimizer = self.optimizer, loss = self.loss)



	# okay, so we begin the functions here
	def train(self, epochs = None, shuffle=True, callbacks = default_callbacks, get_weights=False):
		if epochs is None:
			epochs = self.epochs
		print "Model training:"
		history = self.model.fit(self.input_data, self.output_data, epochs=epochs, shuffle = shuffle, callbacks = callbacks)
		print "Training complete"
		if get_weights:
			weights, biases= self.model.layers[-2].get_weights()
			print weights
			print biases
			return (history, weights, biases)
		return history

			

	def predict(self, test_data = None):
		if test_data is not None:
			return self.model.predict(test_data)
		if test_data is None:
			return self.model.predict(self.test_input)

	def plot_results(self, preds = None, inputs=None, N = 10, start = 0):
		if preds is None:
			preds = self.predict()
		if inputs is None:
			inputs = self.test_output

		print preds.shape
		shape = (preds.shape[1],preds.shape[2])
			
		fig = plt.figure(figsize=(20,4))
		for i in range(N):
			#display original
			ax = plt.subplot(2,N,i+1)
			plt.imshow(inputs[start + i].reshape(shape))
			plt.gray()
			plt.title('original')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			#display reconstructoin
			ax = plt.subplot(2, N, i+1+N)
			plt.imshow(preds[start + i].reshape(shape))
			plt.gray()
			plt.title('reconstruction')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	
		plt.show()
		return fig
		
	
	def get_error_maps(self, input_data = None, predictions= None, return_preds = False):
		# we get all combinations of things here for behaviour
		# we also reshape them here, for the rest of all time, so my functoins are easy

	#okay, this function at the moment is just totally wrong. let's rewrite
		if input_data is None:
			input_data = self.test_input
		if predictions is None:
			predictions = self.predict(test_data = input_data)
		maps = np.absolute(predictions - self.test_output)
		assert input_data.shape == predictions.shape, 'predictoins and input data must have same dimensions'
		shape = predictions.shape

		if return_preds:
			return predictions, np.reshape(maps,(shape[0], shape[1], shape[2]))
		return np.reshape(maps, (shape[0], shape[1], shape[2]))

		"""
		
		#shape = (self.shape[0], self.shape[1], self.shape[2])
		if input_data is None and predictions is None:
			maps= np.absolute(self.test_input - self.test_output)
			#print "test input"
			##print self.test_input.shape
			#print "test output"
			##print self.test_output.shape
			shape = self.test_input.shape
			#print "shape!" + str(shape)
			return np.reshape(maps, (shape[0], shape[1], shape[2]))
		if input_data is None and predictions is not None:
			maps = np.absolute(self.test_input - predictions)
			shape = self.test_input.shape
			return np.reshape(maps, (shape[0], shape[1], shape[2]))
		if input_data is not None and predictions is None:
			maps = np.absolute(input_data - self.test_output)
			shape = self.test_input.shape
			return np.reshape(maps, (shape[0], shape[1], shape[2]))
		if input_data is not None and predictions is not None:
			maps = np.absolute(input_data - predictions)
			shape = self.test_input.shape
			return np.reshape(maps, (shape[0], shape[1], shape[2]))
		"""

	


	def plot_error_maps(self, error_maps = None, N = 10, original_images = None, predictions = None):
		if error_maps is None:
			error_maps = get_error_maps()
		if original_images is None:
			shape = self.test_input.shape
			original_images = np.reshape(self.test_output, (shape[0], shape[1], shape[2]))
		
		if predictions is None:
			for i in xrange(N):
				compare_two_images(original_images[i], error_maps[i], 'Original', 'Error Map')
		if predictions is not None:
			for i in xrange(N):
				imgs = (original_images[i], predictions[i], error_maps[i])
				titles = ('Original','Prediction','Error Map')
				compare_images(imgs, titles)


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
		self.plot_error_maps(mean_maps, N)
		

		
