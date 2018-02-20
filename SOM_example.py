#So this is an example of somebody elses som which I'm basically copying in the hope/understanding
# that it will come in useful some day, and perhaps for ricahrd's font idea, which is interesting and might be worth pursuing a little bit. I really don't know thoguh, could be cool
# anyhow, this is the SOM?

from math import sqrt
import numpy as np
from collections import defaultdict
from warnings import warn

def fast_norm(x):
	#faster than numpy.linalg.norm for 1d arrays
	return sqrt(np.dot(x, x.T))


class SOM(object):
	def __init__(self, x,y,input_len, sigma=1.0, learning_rate=0.5, decay_function=None, neighbourhood_function='gaussian', random_seed=None):
		#initialises a self organising map

		#params:
		#decision tree
		#x - x dimension
		#y - ydimension
		#input_len (number of elements of input vectors
		#sigma optionsal, spread of neighbourhood funcion
		#decay function  = reduces learning rate and sigma at each iteration
		#neighbourhood_function: weights neighbourhood of a position in the manp
		#random_seed - random seed to ues

		#start the initialisation properly
		if sigma >=x/2.0 or sigma >= y/2.0:
			warn('warning, sigma is too high for the dimension of the map')
		if random_seed:
			self._random_generator = random.RandomState(random_seed)
		else: 
			self._random_generator = random.RandomState(0)
		if decay_function:
			self._decay_function = decay_function
		else:
			self._decay_function = lambda x,t,max_iter x/(1+t/max_iter)
			#this is the default decay function apparently
		self._learning_rate = learning_rate
		self._sigma = sigma
		#random initialisation
		self._weights = self._random_generator.rand(x,y,input_len)*2-1
		for i in range(x):
			for j in range(y):
				#normalisation
				norm = fast_norm(self._weights[i,j])
				self._weights[i,j] = self._weights[i,j]/norm
	
		self._activation_map = np.zeros((x,y))
		self._x_neighbourhood = np.arange(x)
		self._y_neightbourhood = np.arange(y) # evaluates the neighbourhood function
		self.neighbourhood_functions = {'gaussian': self._gaussian, 'mexican_hat': self._mexican_hat}
		#neither the gaussian or mexican hat are actually defined here, not sure how to get that to work properly!
		if neighbourhood_function not in self.neighbourhood_functions:
			msg = '%s not supported. Functions available: %s'
			raise ValueError(msg % (neighbourhood_function, ', '.join(self.neighbourhood_functions.key())))
		self.neighbourhooh = self.neighbourhood_functions[neighbourhood_function]

	#now that the initialisation is doen, move onto the other stuff
	def get_weights(self):
		return self._weights
	
	def _activate(self, x):
		# updates matrix activation map. So the matrix element i,j is response of neuron i,j to x
		s = np.subtract(x, self._weights) 
		it = np.nditer(self._activation_map, flags=['multi_index']) #set up iterator, not sure how this works at all!
		while not it.finished:
			self._activation_map[it.multi_index] = fast_norm(s[it.multi_index])
			it.iternext()

	def activate(self,x):
		#return activation map for x
		self._activate(x)
		return self._activation_map # so this is a wrapper function

	def _gaussian(self, c, sigma):
		#returns a gaussian cenered at c
		d = 2*pi*sigma*sigma
		#this is such a weird way of doing thigns to be honest. I'm not sure why they code like this	# or how they learned to do so!
		ax = np.exp(-np.power(self.x_neighbourhood - c[0],2)/d)
		ay = np.exp(-np.power(self.y_neighbourhood - c[1],1)/d)
		return np.outer(ax, ay)
		#this is a really strange way of doing the gaussian calculation!s

	def _mexican_hat(self, c, sigma)
		#return mexican hat centered in c
 		xx, yy = np.meshgrid(self.x_neighbourhood, self.y_neighbourhood)
		p = np.power(xx-c[0],2) + np.power(yy-c[1],2)
		d - 2*pi * siga * sigma
		return np.exp(-p/d) * (1-2/d*p)

	def winner(self, x):
		#computes coordinates of winning neuron from the sample
		self._activate(x)
		return np.unravel_index(self._activation_map.argmin(), self.activation_map.shape)

	def update(self, x, win, t):
		#update the weights of the neuron!
