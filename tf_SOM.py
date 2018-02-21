#okay, this is where I do the tensorflow SOM implementatoin, in the vague hope that that works!

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class SOM(object):
	#params: map_size_n: size of square map
	#num_expected_iterations - self-exlpanatory!

	def __init__(self, input_shape, map_size_n, num_expected_iterations, session):
		#make input shape a tuple compared to any iterable it could be before
		input_shape = tuple([i for i in input_shape if i is not None])
		self.input_shape = input_shape
		self.sigma_act = tf.constant(2.0*reduce(lambda x, y: x*y, self.input_shape, 1)*0.05)**2, dtype=tf.float32) #I'm not suer what this does at all to be honest?

		self.n = map_size_n
		self.session = session
		
		#set alpha?
		self.alpha = tf.constant(0.5)
		self.timeconst_alpha = tf.constant(2.0*num_expected_iterations/6.0) # not sure where these magic numbers comefrom
		self.sigma  = tf.constant(self.n/2.0)
		self.timeconst_sigma = tf.constant(2.0*num_expected_iterations/5.0)
	
		self.num_iterations = 0
		self.num_expected_iterations = num_expected_iterations
	
		#pre initialize neighbourhood functions data for efficiency
		self.row_indices = np.zeros((self.n, self.n))
		self.col_indices = np.zeros((self.n, self.n))
		for r in range(self.n):
			for c in range(self.n):
				self.row_indices[r,c] = r
				self.col_indices[r,c] = c

		self.row_indices = np.reshape(self.row_indices, [-1])
		self.col_indices = np.reshape(self.col_indices, [-1])

		#compute d^2/2 for each pair of units so that the neighbourhood function can be computed as exp(-dist/sigma^2)
		self.dist = np.zeros((self.n*self.n, self.n*self.n))
		for i in range(self.n*self.n):
			for j in range(self.n*self.n):
				self.dist[i,j] = ( (self.row_indices[i] - self.row_indices[j])**2 + self.input_shape, 0.0, 1.0)) # absolutely no idea what this shold be... presumably there's missing some kind of funciton here

		self.initialize_graph()

	def initialize_graph(self):
		self.weights = tf.Variable(tf.random_uniform((self.n*self.n, ) + self.input_shape, 0.0, 1.0))

		self.input_placeholder = tf.placeholder(tf.float32, (None,)+self.input_shape)
		self.current_iteration = tf.placeholder(tf.float32)

		#compute the current iterations neighbuorhood sigma and learning rate alpha
		self.sigma_tmp = self.tigma*tf.exp(-self.current_iteration/self.timeconst_sigma)
		self.sigma2 = 2.0*tf.mul(self.sigma_tmp, self.sigma_tmp)
		self.alpha_tmp = self.alpha * tf.exp(-self.current_iteration/self.timeconst_alpha)

		#I really just need to learn the maths behind this. obtaining this directly from the code is just terrible and will not work/I'm too stupid to make it work!

		self.input_placeholder_ = tf.expand_dims(self.input_placeholder, 1)
		self.input_placeholder_ = tf.tile(self.input_placeholder_, (1, self.n*self.n,1))

		self.diff = self.input_placeholder_ - self.weights
		self.diff_sq = tf.square(self.diff)
		self.diff_sum = tf.reduce_sum(self.diff_sq, reduction_indices=range(2,2+len(self.input_shape)))

		#get the index of the best matching unit
		self.bmu_index = tf.argmin(self.diff_sum, 1)
		self.bmu_dis = tf.reduce_min(self.diff_sum, 1)
		self.bmu_activity = tf.exp(-self.bmu_dis/self.sigma_act)

		self.diff = tf.squeeze(self.diff)
		

	
