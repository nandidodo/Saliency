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
	return (1/np.sqrt(2*math.pi)) * np.exp(-((u - g(v))**2)/2)

print calculate_likelihood(np.sqrt(3),3)
def calculate_prior(v):
	return (1/np.sqrt(2*math.pi)) * np.exp(-((v-3)**2)/2)
print calculate_prior(3)

def calculate_unnormalised_posteriors(vs):
	probs = []
	for v in vs:
		prob = calculate_likelihood(v,u) * calculate_prior(v)
		probs.append(prob)
	return probs

def normalise_probs(probs):
	probs = np.array(probs)
	print len(probs)
	total = sum(probs)
	normed = []
	for i in xrange(len(probs)):
		print "adding prob" + str(i) + " out of " + str(len(probs))
		normed.append(probs[i]/total)
	normed = np.array(normed)
	return normed


print "starting"
vs = generate_vs(0.01, 5, 0.01)
print "generated vs"
probs = calculate_unnormalised_posteriors(vs)
print type(probs)
print len(probs)
#print probs
print probs[3]
print "calculated unnormalised"
normed = normalise_probs(probs)
print "calculated normalised"
#p#rint normed
print probs
plt.plot(normed)
plt.show()

# YAY! it works. it can compute this without silly bugs and is awesome. I finally did some straightforward bayesian inference. that's great!!! should be easy, but small steps tbh

