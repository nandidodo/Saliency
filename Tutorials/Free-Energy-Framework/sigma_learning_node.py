# okay, this is where we do our own python implementation of how this works
# andthesimulation which is meant to learn sigma over time for a single node, before generalising to the multivariate case

import numpy as np
import matplotlib.pyplot as plt

mean_phi = 5
sigma_phi = 2
phi_above = 5
DT = 0.01
MAXT =20
TRIALS = 1000
LRATE = 0.01

# basically the goal here is to update the sigma much less rapidly than the actual experience
# so the error units and so forth have the chance to converge with a single sigma
#before the update occurs, which is interesting, but I honestly don't know if this is the best approach
# if this works then we can try it without the inner loop to see how that fucntions
#like the equivalent, batch norm does the learning after every trial!?

sigma = [1] #initialise weight
for trial in xrange(TRIALS):
	err = [1] # we initialise the error of the prediction error unit
	e = [1]		#inhibitory interneuron
	phi = np.random.normal(mean_phi, np.sqrt(sigma_phi))
	for i in xrange(int(MAXT/DT)):
		pred_err = err[i] + DT * (phi-phi_above - e[i])
		err.append(pred_err)
		inhib = e[i] + DT* ((sigma[trial] * err[i]) - e[i])
		e.append(inhib)
	sigma.append(sigma[trial] + LRATE * (e[-1]*err[-1] -1))

plt.plot(sigma)
plt.show()

# well, that's damn impressive! that maths here works... That's realy really really awesome!
# I'm not sure how to solve that. We can try it in the very simple multivariace case to see what's up
# bu I honestly do not know... first let's try it without the inner loop
