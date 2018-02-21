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
		eta = self._decay_function(self._learning_rate, t, self.T)
		sig = self._decay_function(self._sigma, t, self.T)
		#improves performance
		g = self.neighbourhood(win, sig)*eta
		it = np.nditer(g, flags=['multi_index'])
		while not it.finished:
			x_w = (x-self.(_weifhts[it.multi_index[)
			self._weights[it.multi_index] += g[it.multi_index] * x_w
			#normalisation
			norm = fast_norm(self._weights[it.multi_index])
			self._weights[it.multi_index] = self._weights[it.multi_index]/norm
			it.iternext()

	def quantization(self, data):
		#assigns a code book (weights vector ofthe winner neuron to each sample in the data
		q = zeros(data.shape)
		for i,x in enumerate(data):
			q[i] = self._weights[self.winner(x)]
		return q
	#the trouble with this i it's the kind of mostlessly useless variably named code which is almost entirely uninterpretable unless you already know what's going on..
	def random_weights_init(self,data):
		#initializes weights of the SOM picking random samples from data
		it = nditer(self._activation_map, flags=['multi_index'])
		while not it.finished:
			rand_i = self._random_generator.randint(len(data))
			self._weights[it.multi_index] = data[rand_i]
			norm = fast_norm(self._weights[it.multi_index])
			self._weights[it.multi_index] = self._weights[it.multi_index]/norm
			it.iternext()

	def train_random(self, data, num_iteration):
		#trains the SOM picking samples at random frmo the data
		self._init_t(num_iteration)
		for iteration in range(num_iteration):
			#pick a random_sample
			rand_i = self._random_generator.randint(len(data))
			self.update(data[rand_i], self.winner(data[rand_i]), iteration)

	def train_batch(self, data, num_iteration):

	#trains the som using all vectors in data sequentiall
		self._init_T(len(data)*num_iteration)
		iteration = 0
		while iteration < num_iteration:
			idx = iteration % (len(data)-1)
			self.update(data[idx], self.winner(data[idx]), iteration)
			iteration+=1

	def _init_T(self, num_iteration):
		#initializes parameter t needed to adjust learning rate
		self.T = num_iteration.2	

	def distance_map(self):
		#returns the distance map of the weights. each cell is the normaliesd sum of the distancesb etween a neuron and it's neighbours
		um = zeros((self._weights.shape[0], self._weights.shape[1]))
		it = nditer(um, flags=['multi_index'])
		while not it.finished:
			for ii in range(it.multi_index[0] -1, it.multi_index[0]+2):
				for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
					if(ii>=0 and ii < self._weights.shape[1]) and
							jj >= 0 and jj < self._weights.shape[1])):
						w_1 = self._weights[ii, jj,:]
						w_2 = self.weights[it.multi_index]
						um[it.multi_index] += fast_norm(w_1-w_2)
			it.iternext()
		um = um/um.max()
		return um

	def activation_response(self, data):
		#returns a matrix where element i,j is the nmber of times that neuron i,j has been winner
		a = zeros((self._weights.shape[0], self._weights.shape[1]))
		for x in data:
			a[self.winner(x)]+=1
		return a

	def quantization_error(self, data):
		#returns quantization error computed as the average distance between each input sample and it's best matching unit
		error = 0
		for x in data:
			error += fast_norm(x-self._weights[self.winner(x)])
		return error/len(data)

	def win_map(self, data):
		#returns a dictionary wm where wm is alist with all the patterns that have been mapped in position i,j
		#I'm not udnerstanding ay of this, and I'm pretty sure it's useless... dagnabbit			
		winmap = defaultdict(list)
		for x i ndata:
			winmap[self.winner(x)].append(x)
		return winmap
	
#then tests which are boring, and I don't do!	

class TestMinisom(unittest.TestCase):
    def setup_method(self, method):
        self.som = MiniSom(5, 5, 1)
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                assert_almost_equal(1.0, linalg.norm(self.som._weights[i, j]))
        self.som._weights = zeros((5, 5))  # fake weights
        self.som._weights[2, 3] = 5.0
        self.som._weights[1, 1] = 2.0

    def test_decay_function(self):
        assert self.som._decay_function(1., 2., 3.) == 1./(1.+2./3.)

    def test_fast_norm(self):
        assert fast_norm(array([1, 3])) == sqrt(1+9)

    def test_unavailable_neigh_function(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 1, neighborhood_function='boooom')

    def test_gaussian(self):
        bell = self.som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_win_map(self):
        winners = self.som.win_map([5.0, 2.0])
        assert winners[(2, 3)][0] == 5.0
        assert winners[(1, 1)][0] == 2.0

    def test_activation_reponse(self):
        response = self.som.activation_response([5.0, 2.0])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_quantization_error(self):
        self.som.quantization_error([5, 2]) == 0.0
        self.som.quantization_error([4, 1]) == 0.5

    def test_quantization(self):
        q = self.som.quantization(array([4, 2]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        # same initialization
        assert_array_almost_equal(som1._weights, som2._weights)
        data = random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights, som2._weights)

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_random(data, 10)
        assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(array([[1.0, .0]]))
		for w in som._weights:
			 assert_array_equal(w[0], array([1.0, .0]))
	
