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
TRIALS = 1000
MAXT = 1000
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
		#if sig <0:
		#	sig = 2
		sigma.append(sig)
		print err[-1]
		print e[-1]
		print sigma[-1]
		print "  "

	plt.plot(sigma)
	plt.show()



# the transition to matrices is basically entirely straightforward
TRIALS = 1000
MAXT = 1
mean_phi = [5,5]
upper = [5,5]
sigma_phi = [[2,0],[0,2]]
def multidimensional_with_inner_loop():
	sigma = [[1,0],[0,1]]
	sigma = np.array(sigma)
	print sigma.shape
	for trial in xrange(TRIALS):
		print "in loop " + str(trial)
		err = [[1,1]]
		err = np.array(err)
		#print err.shape
		e = [[1,1]]
		e = np.array(e)
		#print e.shape
		phi = np.random.multivariate_normal(mean_phi, np.sqrt(sigma_phi))
		for i in xrange(int(MAXT/DT)):
			pred_err = err[i] + LRATE * (phi-phi_above - e[i])
			pred_err = [pred_err]
			pred_err = np.array(pred_err)
			#print pred_err.shape
			err = np.concatenate((err, pred_err),axis=0)
			inhib = e[i] + LRATE * ((sigma[trial] * err[i]) - e[i])
			inhib = [inhib]
			inhib = np.array(inhib)
			e = np.concatenate((e, inhib),axis=0)

		#plt.plot(err)
		#plt.plot(e)
		#plt.show()
		sig = sigma[trial] + LRATE * (np.dot(e[-1].T,err[-1].T) -1) # this -1 here has a huge impact on the algorith, but I don't know how to deal wih it!?
		#if sig <0:
		#	sig = 2
		sig = [sig]
		sig = np.array(sig)
		sigma = np.concatenate((sigma, sig), axis=0)
		print err[-1]
		print e[-1]
		print sigma[-1]
		print "  "

	s = sigma[:,0]
	for i in xrange(len(s)):
		print s[i]
	
	print len(s)
	plt.plot(sigma[:,0])
	#plt.plot(sigma[1])
	plt.show()

# okay, yeah, this is numerically unstable, and we're just going to have to deal with that instability, I fear. oh well, such is life. we'll look more at this tomorrow!
	
	
# yep, the multidimensional case seems even more insane than the nonmultidimensinoal, and I don't understnad it at all to be honest
#multidimensional_with_inner_loop()

#I just straight up don't understand why this doesn't work and why it is so horrendously unstable numerically. It just sucks... argh! Could we do everything with logs?
# yep, I mean dimensioanlly this is horrendously unstable and diverges like nobody's business
# I'm really not sure how stability of this kind of system actually works under any sort of reasonable thing. I'm not sure if I'm doing something wrnog, because it seems to have all sorts of truly insane feedback loops everywhere and I don't know what to od about it... dagnabbit!
#okay, for some really strange reason this thing just straight up doesn't work, and I don't know why this system diverges so constantly... dagnabbit. But for some reason this time the values just get bounced around one after the other, and I don't understand why each time the values are flipped... argh?


# so the problem is that it appears to be horrifically numerically unstable if we don't have an inner loop in that we need to get the two to stabilise before we update the sigma, but that's just really annoying and I don't udnerstand why we need that. Because it's just stupid to be honest
# the only other thing is perhaps if we destroy the sigma learning rate a lot it could help to stop the blowup

# yeah, this thing is really just incredibly numerically unstable. and it works sometimes but I don't understand why it is numerically unstable to be hoenst, and why the solution is just a seemingly endless decrease in some cases? I really do not know?
# I think it's just that if it goes negative, it can never come back up
# okay, there's randomness in the algorith, but there seems to be a tipping point around 2, at which point it diverges in either direction and never comes back? and I just don't know why to be honest? the trouble is that it generally seems to ahve really that this straight up doesn't ever converge, which is pretty bad, and that the error units and stuff oscillate too damn much for it to be reasonable, and I don't know hw the brain does dampning. like seriously, it is a big problem. I wonder if the multidimensional version will be calmer or crazier -- most likely crazier!
# basically I think the problem is that these things need a time to stabilise around the correct value, but I don't know why. why do e and err need to stabilise about a single value while the rest does not. That is kind of annoying and difficult numerically. I don't see why it diveges negatively so rapidly?

#run_without_inner_loop()
#let's just try the multidimensional case and see if this helps us or hurts usto be honest
# then if this actually works reasonably, we will have all the building blocks we need to actually start creating a model, although concatenating even more of these things together I'm sure is just an utter recipe for chaos, disaster and numerical instability on an enormous scale as they are all just insane coupled oscillators with a whole myriad of feedback loops that cannot be constrained, and really we don't know how the brain is meant to handle this tbh, apart from strict neurological limits, but yeah, our math systems are amazingly fragile to feedback loops while in the real world basiacally everything is kept withinn sane sort of parameters!so idk how that works! I suppsoe it's due to some kind of sigmoid of exponentially increasing difficulty or whatever... I really do nt know!





