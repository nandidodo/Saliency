# okay, a simple bayesian inference approach... let's do this - I'm actually relatively unpracticed at even simple 1D bayesian inference which is really bad. so let's write the problem specification and figure it out
#http://www.sciencedirect.com/science/article/pii/S0022249615000759?via%3Dihub#f000030
#problem specification:
# we have animal trying to detect scalar size of food. Size is v. Light intensity observation is u. Related by u = g(v) = v^2. 
# We have prior v ~ N(3,1). And likelihood p(u|v) ~N(u;g(v), 1)
#we see u = 2. What is likely v from observation?
#so we apply bayes p(v|u) ~=p(u|v)p(v)
# and plot. fariyl simple

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import math

#vs = np.linspace(0.001, 5, step=0.001)
#print len(vs)
u=2

#we do this the old fashioned way
def generate_vs(start, stop, step):
	vs = []
	vs.append(start)
	curr = start
	while curr < stop:
		curr += step
		vs.append(curr)
	return vs

def g(v):
	return v**2

def calculate_likelihood(v,u):
	return (1/np.sqrt(2*math.pi)) * np.exp(-(u - g(v)**2)/2)

def calculate_prior(v):
	return (1/np.sqrt(2*math.pi)) * np.exp(-((v-3)**2)/2)

def calculate_unnormalised_posteriors(vs):
	probs = []
	for v in vs:
		prob = calculate_likelihood(v,u) * calculate_prior(v)
		probs.append(probs)
	return probs

def normalise_probs(probs):
	probs = np.array(probs)
	total = sum(probs)
	normed = []
	for prob in probs:
		normed.append(probs/total)
	normed = np.array(normed)
	return normed

vs = generate_vs(0.01, 5, 0.01)
probs = calculate_unnormalised_posteriors(vs)
normed = normalise_probs(probs)
plt.plot(normed, probs)
plt.show()
	

