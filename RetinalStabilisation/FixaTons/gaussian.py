# just quickly findthe mle's of gaussian data for this

from __future__ import division
import numpy as np 


def MLE_mu(data):
	N = len(data)
	return np.sum(data)/N

def MLE_sigma_squared(data, return_mean=True):
	N = len(data)
	mu = MLE_mu(data)
	#loop to calculate this
	total = 0
	for i in xrange(N):
		total += (data[i] - mu)**2
	variance = total/N
	if return_mean:
		return variance, mu
	return variance

def gaussian_probability(x, mu, sigma):
	denom = 1/(np.sqrt(2*np.pi * np.sqrt(sigma)))
	num = -1* np.square(x-mu)/(2*sigma)
	return denom * np.exp(num)

def gaussian_probabilities(data):
	variance, mu = MLE_sigma_squared(data, return_mean=True)
	#std = np.sqrt(variance)
	ps = []
	for i in xrange(len(data)):
		ps.append(gaussian_probability(data[i], mu, variance))
	ps = np.array(ps)
	return ps

