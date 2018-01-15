# okay, another simulation finding the same solution by gradient ascent

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#constants for our problem
sp = 1
su = 1

def g(v):
	return v**2
def gdash(v):
	return 2*v

vp = 3
u= 2

def calculate_gradient(phi):
	return ((phi - vp)/sp) + (((u-g(phi))/su)*gdash(phi))

def gradient_update(oldphi, step):
	return oldphi + step*calculate_gradient(oldphi)

def gradient_ascent(start, num, lrate=0.15):
	results = []
	oldphi = gradient_update(start,lrate)
	results.append(oldphi)
	for i in xrange(num-1):
		oldphi = gradient_update(oldphi, lrate)
		results.append(oldphi)
	return results


results = gradient_ascent(vp, 50)
print results
plt.plot(results)
plt.show
