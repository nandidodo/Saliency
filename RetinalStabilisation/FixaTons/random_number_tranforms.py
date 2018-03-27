# okay, some tranformation functions to allow sampling from various distibutions
# by the transform method
# which involves sampling from a U[0,1] and then transforming the output according
#to some equatoino

from __future__ import division
import numpy as np

def uniform_sample(low=0, high=1):
	return np.random.uniform(low=low, high=high)

def power_law(xmin, alpha):
	u = uniform_sample()
	return xmin*(1-u)**(-1/(alpha-1))

def exponential(xmin, lamba):
	u = uniform_sample()
	return xmin - (1/lamba)*np.log(1-u)


def stretched_exponential(xmin, l, beta):
	u = uniform_sample()
	const = xmin**beta
	var = (1/l) * np.log(1-u)
	return (const - var)**(1/beta)

def log_normal(sigma):
	u1 = uniform_sample()
	u2 = uniform_sample()
	p = np.sqrt(-2*np.square(sigma) * np.log(1-u1))
	theta = 2 * np.pi * u2
	x1 = np.exp(p*np.sin(theta))
	x2 = np.exp(p*np.cos(theta))
	return x1, x2

	
