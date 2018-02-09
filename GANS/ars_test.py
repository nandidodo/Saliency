#okay, so this is not going to actually have anything to do with GAns
# just a standard NN function approxiamtino. I'm pretty sure smoebody must have done this
#before. So I shall search. If not it could be helpful perhaps to see just how good 
#NNs can be at this. It should be relatively straightforward to test
# so the aim here should be to see if the nn can actuallyl earn the label function in a straightfoward manner
# hopefully it can but I'm not sure. I should test it out with a simple gaussian
# or something first to get an idea

from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


input_dims = 1
intermediate_dim = 10
output_dims = 1
def testnet(z):
	with tf.variable_scope('network'):
		#intermediate_dim = 10

		W1 = tf.get_variable('w1', [input_dims, intermediate_dim], initializer=tf.random_normal_initializer(stddev=0.1))
		B1 = tf.get_variable('b1', [intermediate_dim], initializer=tf.constant_initializer())

		W2 = tf.get_variable('w2', [intermediate_dim, intermediate_dim], initializer=tf.random_normal_initializer(stddev=0.1))
		B2 = tf.get_variable('b2', [intermediate_dim], initializer=tf.constant_initializer())

		W3 = tf.get_variable('w3', [intermediate_dim, output_dims], initializer=tf.random_normal_initializer(stddev=0.1))
		B3 = tf.get_variable('b3', [output_dims], initializer=tf.constant_initializer())


		fc1 = tf.nn.relu(tf.matmul(z,W1) + B1)
		#not sure what activation functoin... I definitely don't want probabilities/sigmoids here
		fc2 = tf.nn.relu(tf.matmul(fc1, W2) + B2)
		fc3 = tf.abs(tf.matmul(fc2, W3) + B3)

		return fc3 # can't think of anything better to do

		# I'm gonig to try it with 1 D gaussian!

def prob_func(x, params):
		#do something here to return a vector of correct dim size)
		#for now try it with a single univariate gaussian
		#so I just need to implement the gaussian equation here
		#mu, sigma = params
		#return (1/tf.sqrt(2*math.pi * sigma**2)) * tf.exp(-((x-mu)**2)/2*sigma**2)

		#let's try a simpler one - a unifomr function
		return tf.convert_to_tensor(1/20)
def euclid_distance_loss_func(x,y):
	#assert len(x) == len(y),'network and probability function have different dimensions!'
	return tf.sqrt(tf.reduce_sum(x-y)**2)


def plot_sample_acceptance_rate(samples,bin_width):
	avgs = []
	for i in xrange(len(samples)//bin_width):
		samps = samples[i*bin_width: (i+1)*bin_width]
		avg = np.mean(samps)
		avgs.append(avg)
	avgs = np.array(avgs)
	plt.plot(avgs)
	plt.title('plot sample acceptance rate over bin widths')
	plt.xlabel('bin')
	plt.ylabel('average acceptance rate')
	plt.show()

def percent_acceptances(acceptances, rejects):
	return (acceptances/(acceptances + rejects)) * 100

def plot_sample_histogram(samples, bins=100):
	print type(samples)
	print samples.shape
	plt.hist(samples, bins, normed=1, facecolor="blue")
	plt.xlabel("x")
	plt.ylabel("Probability")
	plt.title('histogram')
	plt.show()

def plot_losses(losses):
	plt.plot(losses)
	plt.title('loss values over time')
	plt.xlabel('iteration')
	plt.ylabel('loss value')
	plt.show()


def train_and_sample():
	input_dim = 1
	output_dim = 1
	num_samples = 100000
	epochs = 10
	mu = 0
	sigma = 1
	#samples= np.random.uniform(low=-10, high=10, size=num_samples)
	params = [mu, sigma]
	runs = 10000
	losses = []
	accept_rejects = []

	num_acceptances = 0
	num_rejections = 0

	lr = 0.0001

	with tf.variable_scope('placeholder'):
		X = tf.placeholder(tf.float32, [None,1])
		tf.summary.histogram('input', X)

	# at the moment this will only work when everythign is single dimensional
	# so I should expect crazy gradient updates?
	#define loss
	with tf.variable_scope('result'):
		res = testnet(X)

	with tf.variable_scope('loss'):
		loss = euclid_distance_loss_func(testnet(X), prob_func(X, params))
	
	with tf.variable_scope('train'):
		train_step = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

	with tf.variable_scope('prob'):
		prob_height = prob_func(X, params)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	samples = []
	#nex


	for i in xrange(epochs):
		for j in xrange(runs):
			sample = np.random.uniform(low=-10, high=10,size=1)
			height = np.random.uniform(low=0, high=1, size=1)
			sample = np.reshape(sample,(1,1))
			result, loss_val, _ ,actual_height= sess.run([res, loss, train_step, prob_height], feed_dict={X: sample})
			sample_height = result*height
			print "Epochs: " + str(i) + "runs: " + str(j) + "loss: " + str(loss_val) + " Sample: " + str(sample)
			print "Result: " + str(result) + "Actual height: " + str(actual_height)
			#now for the sampling step
			losses.append(loss_val)
			if sample_height <= actual_height:
				#this means acceptance
				samples.append(sample)
				num_acceptances +=1
				print "ACCEPTED"
				accept_rejects.append(1)
			if sample_height > actual_height:
				#rejection, discard samlpe
				num_rejections +=1
				print "Rejected"
				accept_rejects.append(0)

	#not sure what to do now as mycah's music is really distracting... dagnabbit!

	# I need to plot all the samples presumably

	samples = np.array(samples)
	samples = np.reshape(samples, (len(samples)))
	losses = np.array(losses)
	accept_rejects = np.array(accept_rejects)
	plot_sample_histogram(samples)
	print "Percent acceptances: " + str(percent_acceptances(num_acceptances, num_rejections))
	print "Num acceptances: " + str(num_acceptances)
	print "Num rejections: " + str(num_rejections)
	bin_width = 100
	plot_sample_acceptance_rate(accept_rejects, bin_width)
	plot_losses(losses)





if __name__ == '__main__':
	train_and_sample()






	