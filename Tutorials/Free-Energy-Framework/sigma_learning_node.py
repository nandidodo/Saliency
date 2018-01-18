# okay, this is where we do our own python implementation of how this works
# andthesimulation which is meant to learn sigma over time for a single node, before generalising to the multivariate case

import numpy as np
import matplotlib.pyplot as plt

mean_phi = 5
sigma_phi = 2
phi_above = 5
DT = 1
MAXT =1000
TRIALS = 1000
LRATE = 0.01

# basically the goal here is to update the sigma much less rapidly than the actual experience
# so the error units and so forth have the chance to converge with a single sigma
#before the update occurs, which is interesting, but I honestly don't know if this is the best approach
# if this works then we can try it without the inner loop to see how that fucntions
#like the equivalent, batch norm does the learning after every trial!?
def run_with_inner_loop():
	sigma = [1] #initialise weight
	for trial in xrange(TRIALS):
		err = [1] # we initialise the error of the prediction error unit
		e = [1]		#inhibitory interneuron
		phi = np.random.normal(mean_phi, np.sqrt(sigma_phi))
		for i in xrange(int(MAXT/DT)):
			pred_err = err[i] + LRATE * (phi-phi_above - e[i])
			err.append(pred_err)
			inhib = e[i] + LRATE* ((sigma[trial] * err[i]) - e[i])
			e.append(inhib)

		sigma.append(sigma[trial] + LRATE * (e[-1]*err[-1] -1))

	plt.plot(sigma)
	plt.show()

# well, that's damn impressive! that maths here works... That's realy really really awesome!
# I'm not sure how to solve that. We can try it in the very simple multivariace case to see what's up
# bu I honestly do not know... first let's try it without the inner loop
TRIALS = 50000
MAXT = 1
LRATE = 0.01
mean_phi = 5
phi_above=5
def run_without_inner_loop():
	sigma = [1] # init sigma

	for trial in xrange(TRIALS):
		print "in loop " + str(trial)
		err = [1]
		e = [1]
		phi = np.random.normal(mean_phi, np.sqrt(2))
		for i in xrange(int(MAXT/DT)):
			pred_err = err[i] + LRATE * (phi-phi_above - e[i])
			#print "past pred err"
			err.append(pred_err)
			inhib = e[i] + LRATE * ((sigma[trial] * err[i]) - e[i])
			e.append(inhib)
		#plt.plot(err)
		#plt.plot(e)
		#plt.show()
		sig = sigma[trial] + LRATE * (e[-1] * err[-1] -1) # this -1 here has a huge impact on the algorith, but I don't know how to deal wih it!?
		if sig <0:
			sig = 2
		sigma.append(sig)
		print err[-1]
		print e[-1]
		print sigma[-1]
		print "  "

	plt.plot(sigma)
	plt.show()

# so the problem is that it appears to be horrifically numerically unstable if we don't have an inner loop in that we need to get the two to stabilise before we update the sigma, but that's just really annoying and I don't udnerstand why we need that. Because it's just stupid to be honest
# the only other thing is perhaps if we destroy the sigma learning rate a lot it could help to stop the blowup

# yeah, this thing is really just incredibly numerically unstable. and it works sometimes but I don't understand why it is numerically unstable to be hoenst, and why the solution is just a seemingly endless decrease in some cases? I really do not know?
# I think it's just that if it goes negative, it can never come back up
# okay, there's randomness in the algorith, but there seems to be a tipping point around 2, at which point it diverges in either direction and never comes back? and I just don't know why to be honest? the trouble is that it generally seems to ahve really that this straight up doesn't ever converge, which is pretty bad, and that the error units and stuff oscillate too damn much for it to be reasonable, and I don't know hw the brain does dampning. like seriously, it is a big problem. I wonder if the multidimensional version will be calmer or crazier -- most likely crazier!

run_without_inner_loop()



