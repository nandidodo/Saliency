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
phis = [3]
eps = [0]
eus = [0]
u = 2
sigma_u = 1
sigma_p = 1

#what order do we evaluate nodes in... who even knows to be honest?
# I think we should do prediction errors first, but I really don't know
# do we fix u to a single value? let's define sigma u and p here as simple values
num_runs = 100
lr = 0.1

# and our values so we can plot them
#let's just stat by calculating the errors first and then phi hopefully this will be numeriacally stable and not go crazy
# okay, well that doesn't work and is not at all numerically stable... dagnabbit!
#but I haven't actualy implementedthis wrong that I see... f... but it just blows up insanely. dagnabbit!
"""
for i in xrange(num_runs):
	eps.append(e_p)
	e_p = eps[i] + lr*(phi - v_p - (sigma_p*e_p))

	eus.append(e_u)
	e_u = eus[i] + lr*(u - g(phi) - (sigma_u * e_u))

	phis.append(phi)
	phi = phis[i] + lr*(e_u*gdash(phi) * -e_p)
	print "Epoch: " + str(i)
	print (phi, e_u, e_p)
"""

# they do phi first as well. let's copy their thing exactly and hope it works. who knew that making numerically stable examples is so difficult!
for i in xrange(num_runs):
	oldphi = phis[i]
	oldep = eps[i]
	oldeu = eus[i]

	phi = oldphi + (lr * (-oldep + (2*oldphi * oldeu)))
	phis.append(phi)
	
	ep = oldep + (lr * (oldphi - v_p - (sigma_p*oldep)))
	eps.append(ep)

	eu = oldeu + (lr * (u - oldphi**2 - (sigma_u*oldeu)))
	eus.append(eu)

	print "Epoch: " + str(i)
	print (phi, ep, eu)

#maybe that is correct and that's what this is meant to converge to, except I'm very sure it's not
# since it shold be between 2 and 3 but I honestly do not even know to be honest
# I have no ide what I'm doing wrong, so argh! this is converging but to a completely wrong value, and  I don't understand why!
# I mean maybe it is reasonable in some sense? I really have no idea? I mean the idea is that the u is the v squared so it should be between sqrt of two right?
# oh wow! it is actually right! that's awesome!!! we implemented it right!

fig = plt.figure()
plt.plot(phis)
plt.plot(eps)
plt.plot(eus)
plt.show(fig)
