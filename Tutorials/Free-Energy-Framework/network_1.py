# okay, this is meant to be a fairly straightforward implementation of the network,
# this time not wit object oriented stuff as perhaps I should, but I need to see how to extend it
# basically we have a couple of nodes here and I realy don'tk now. There's no connection strength particularly interesting, so idk realy!


import numpy as np
import matplotlib.pyplot as plt

def g(x):
	return x**2

def gdash(x):
	return 2*x

#initialise our constants
v_p =3
phi = v_p
e_p = 0
e_u = 0
u = 2
sigma_u = 1
sigma_p = 1

#what order do we evaluate nodes in... who even knows to be honest?
# I think we should do prediction errors first, but I really don't know
# do we fix u to a single value? let's define sigma u and p here as simple values
num_runs = 1000
lr = 0.01

# and our values so we can plot them
eps = []
eus = []
phis = []

#let's just stat by calculating the errors first and then phi hopefully this will be numeriacally stable and not go crazy
# okay, well that doesn't work and is not at all numerically stable... dagnabbit!
#but I haven't actualy implementedthis wrong that I see... f... but it just blows up insanely. dagnabbit!

for i in xrange(num_runs):
	eps.append(e_p)
	e_p = eps[i] + lr*(phi - v_p - (sigma_p*e_p))

	eus.append(e_u)
	e_u = eus[i] + lr*(u - g(phi) - (sigma_u * e_u))

	phis.append(phi)
	phi = phis[i] + lr*(e_u*gdash(phi) - e_p)
	print "Epoch: " + str(i)
	print (phi, e_u, e_p)

# they do phi first as well. let's copy their thing exactly and hope it works. who knew that making numerically stable examples is so difficult!


