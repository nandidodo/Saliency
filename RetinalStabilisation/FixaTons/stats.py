# okay, aim here will be to figure out and then run what statistical tests
# I need to do to a.)confirm that the results are not power law distributed
# though it is plainly obvious that they are not
# and secondly to figure out what the distributio nactually is
# probably some kind of log normal, I reallydon't know. if so then that would give a good
# narrative of the result - some kind of multiplicative combination of multiple independenteffects
# will result in the log normal distribution as confirmed by the central limit theorem
# so that would be an interesting hypothesis, but I just don't know!

# I'm going to try to read and understand the power law paper to see if anything interesting
# there can occur and see if that helps me figure out exactly what tests I need to run
#http://www.jstor.org/stable/pdf/25662336.pdf?casa_token=m_yCM-YZWRkAAAAA:Gg0B220xc2Z1EkvO9Q5W5346zAbHqGiYZGMuOV_emfxLwYTP0xd-9kfz4Cv75jCQpdrHoJhz_32gZjiwCxpvhOaXkDgCCp3bvoYnRaodzmYExJrFMgNf


# so here is their recipe for checkint it is a power law
# first off - estimate parameters xmin and alpha of power law model 
# then calculate goodness of fit between data and powerlaw using sections 
# check palue else reject
# then compare with alternative hypotheses via likelihood ratio test 
# that's the critical thing but who even knows?
# you cuold use a fully bayesian approach or any  other you want which is really cool
# but this will be simply following the recipes in this paper!


# the trouble here is that I don't know xmin either, but I can theoretically calculte
#xmin probably without any serious issues here hopefully and the MLE
# which is also the hill estimator
# so, the MLE for alpha is alpha_hat = 1 + n * [sum(ln(xi/xmin)))**-1]

#let's write a function to do that sipmply

# THE MLEs are slightly biaed but this bias decreases at rate O(n**-1)
# but this is usually negligible compared to O(n**-1/2) of the standard error
#>=50 datapoints is usually enough for that to be fine

from __future__ import division
import numpy as np
#get gaussian error function erf
from scipy.special import erf
from gaussian import *
from test import *
#from lognormal import *


#returns the probabiliy of a point given the power law
def power_law(x, alpha,xmin):
	norm = (alpha-1)/xmin
	power = (x/xmin)**(-1*alpha)
	if (np.isnan(norm*power)):
		print "Potential nan encountered in power law"
		print "x: ", x
		print "alpha: ", alpha
		print "xmin: " , xmin
		print "norm: " , norm
		print "power: " , power
		return 0
	return norm*power


#hill estimator
def alpha_MLE(data, xmin):
	assert xmin>0, 'Minimum value must be greater than zero'
	data = data_above_xmin(data, xmin)
	N = len(data)
	#do the main sum
	MLE_sum = 0
	for i in xrange(N):
		MLE_sum += np.log(data[i]/xmin)
	return 1 + (N * (1/MLE_sum))


def MLE_error(data,xmin):
	N = len(data)
	a = alpha_MLE(data, xmin)
	error = (a - 1)/np.sqrt(N)
	return a, error

def MLE_error(alpha, N):
	return (alpha-1)/np.sqrt(N)


def power_law_cdf(data, alpha, xmin):
	N = len(data)
	ps = []
	ps_index = -1
	for i in range(N):
		x = data[i]
		if x >=xmin:
			p = power_law(x, alpha, xmin)
			if ps_index>=0:
				p+=ps[ps_index]
				ps.append(p)
				ps_index+=1

	ps = np.array(ps)
	return ps

def power_law_pdf(data, alpha, xmin):
	ps = []
	for i in xrange(len(data)):
		if(data[i])>=xmin:
			ps.append(power_law(data[i], alpha, xmin))
	ps = np.array(ps)
	ps = normalise_distribution(ps)
	return ps

def data_cdf(data,xmin):
	
	pos_data = data_above_xmin(data, xmin)
	N = len(pos_data)
	total = sum(pos_data)
	pos_data = pos_data/total
	for i in range(N-1):
		pos_data[i+1] = pos_data[i] + pos_data[i+1]
	return pos_data

def data_above_xmin(data, xmin):
	pos_data= []
	for i in xrange(len(data)):
		if(data[i]>=xmin):
			pos_data.append(data[i])
	pos_data = np.array(pos_data)
	return pos_data




def KS_test(data, alpha, xmin):
	cdf = power_law_cdf(data, alpha,xmin)
	dist = data_cdf(data, xmin)
	return np.max(np.abs(dist - cdf))


# this i sweighted slightly differently
# t account betterfor extrmeme values near zero or infinity
# which are moredowneighted by the standard ks
# but usually it does not and should not make much difference
# to the overall results!
def weighted_KS(data, alpha, xmin):
	cdf = power_law_cdf(data, alpha, xmin)
	dist = data_cdf(data, xmin)
	#create diffs array by loop!
	diffs = []
	assert len(cdf)==len(dist), 'Distibutions must be of same length'
	for i in range(len(cdf)):
		num = np.abs(cdf[i] - dist[i])
		denom = np.sqrt(dist[i] * (1-dist[i]))
		diff = num/denom
		diffs.append(diff)

	diffs = np.array(diffs)
	return np.max(diffs)


def find_xmin(data, min_xmin=0, max_xmin=None):
# also need a principled way to calculate the xmin so let's figure this out - it uses the minimising
# the distance between power law model and empirical data, so hopefully not too horendous

	if max_xmin is None:
		max_xmin = np.max(data)

		# go through all possible xmins
	best_xmin = 0
	Kss = []
	for xmin in range(min_xmin, max_xmin):
		alpha = alpha_MLE(data, xmin)
		KS = KS_test(data, alpha, xmin)
		Kss.append(KS)
		if KS <=np.min(Kss):
			best_xmin=xmin

	return xmin

def log_likelihood_ratio(dist1, dist2):
	# here the distributions are assumed to simply be 
	#approximated by vectors of probabilities
	assert len(dist1)==len(dist2), 'Distributions must have the same length'
	ratio = 0
	for i in range(len(dist1)):
		ratio += (np.log(dist1[i]) - np.log(dist2[i]))
	return ratio
	# in log space it's a simple sun of subtraction isntead of a product of divisions

def log_likelihood_variance(dist1, dist2):
	N = len(dist1)
	assert N ==len(dist2), 'distributions must have same length'
	#take logs
	dist1 = np.log(dist1)
	dist2 = np.log(dist2)

	#procede as normal!
	mu1 = np.mean(dist1)
	mu2 = np.mean(dist2)
	total = 0
	for i in range(N):
		total += ((dist1[i]-dist2[i])-(mu1-mu2))**2
	return total/N

# I need to begin being able to calculate log likelyhood ratios

def normalised_log_likelihood_ratio(ratio,N,std):
	assert ratio>=0,'Ratio must be positive'
	if ratio==0:
		ratio=-10*100
	ratio = np.log(ratio)
	return np.sqrt(N) * ratio * std

def likelihood_ratio_p_value(ratio, variance, N):
	num = np.abs(ratio)
	denom = np.sqrt(2*np.pi*variance)
	normalised_ratio = num/denom
	p = 1-erf(normalised_ratio)
	return p, ratio




def log_likelihood_test(dist1, dist2):
	print len(dist1)
	print len(dist2)
	assert len(dist1)==len(dist2), 'Distributions compared must have same length'
	N = len(dist1)
	ratio = log_likelihood_ratio(dist1, dist2)
	variance =log_likelihood_variance(dist1, dist2)
	return likelihood_ratio_p_value(ratio, variance, N)


def sample_power_law(N, alpha):
	return np.random.power(alpha, size=N)


# lol! my power law version worked and the horrendous numpy implementation
# did not. That is pretty bad, all things considered
# although possibly due to the utterly tiny xmin messing everything up in their implementation
# although I never pass it in as a parameter, so Ireally don't know!
def sample_power_law_test(N, alpha):
	samps = np.random.uniform(low=0, high=1, size=N)
	return np.power((1-samps), (-1/alpha-1))


def sample_lognormal(N, mu, variance):
	return np.random.lognormal(mu, variance, N)


def sample_lognormal_mine(N, mu, variance):
	samps = []
	length = N//2
	for i in xrange(length):
		u1 = np.random.uniform(low=0, high=1)
		u2 = np.random.uniform(low=0, high=1)
		p = np.sqrt(-2*variance * np.log(1-u1))
		theta = 2 * np.pi * u2
		x1 = np.exp(p*np.sin(theta))
		x2 = np.exp(p*np.cos(theta))
		samps.append(x1)
		samps.append(x2)

	samps = np.array(samps)
	if len(samps)<N:
		diff = N-samps
		for i in xrange(diff):
			u1 = np.random.uniform(low=0, high=1)
			u2 = np.random.uniform(low=0, high=1)
			p = np.sqrt(-2*variance * np.log(1-u1))
			theta = 2 * np.pi * u2
			x1 = np.exp(p*np.sin(theta))
			x2 = np.exp(p*np.cos(theta))
			samps.append(x1)
		samps = np.array(samps)
	return samps


def sample_exponential_mine(N,l):
	samps = np.random.uniform(low=0, high=1, size=N)
	return -1* (1/l) * np.log(1-samps)


def MLE_lambda(data):
	# it's just the reciprocal of the mean!
	return 1/np.mean(data)


def exponential_pdf(data, l):
	ps = []
	for i in xrange(len(data)):
		res = l * np.exp(-1*l*data[i])
		ps.append(res)
	ps = np.array(ps)
	ps = normalise_distribution(ps)
	return ps

def normalise_distribution(dist):
	total = np.sum(dist)
	return dist/total



#something is wrong with the power law atm
# it's meant to give heavier tails, but it obviously does not
# however the pretend gaussian with the MLE statistics gives exactly the right pattern
# but decays too fast. There is a slightly heavier tail then there perhaps hsould be!?
#I've got power law statistica tests
# now I just need todothe same to fit to log normal and exponential and others
# and normal obviosuly and measure degree of fit to be reasnoable!


# okay, now test the likelihood of power law vs standard gaussian
# this is extremely unlikely to work, but if it ooes it would be cool
# and would provide an easy framework for calculating this stuff
# even if I had to hand roll all my own statistics, which isprobably bad!


# the alpha mle calcualted is just aboev 1, meaning a terrifically slow decrease, right?
# so why is the power law sampler failing so utterly!
# it should be generating huge outliers, as it's a heavy tails distribution
# but it is not!

# okay, so it's definietly not a power law OR a log normal, which also appears like
# a standard thing on the log plot
# also for whatever reason the numpy sampling functoins very rarely actually work
# which is kind of funny. Oh wel. They don't work for me, which is just hilarious
# but I don't know. I guess I'm not a numpy implementor, which is wy I don't know this!

# I guess exponential disitribution is the other one to test


# crap, it could easily be the exponential distribution!!! dagnabbit
# that looks more likely, and moer higher tailed than the gaussian. that could cause
# serious issues of analysis, which is quite annoying...dagnabbit!
if __name__=='__main__':
	data, n,bins,patches = get_saccade_distances('fixaton_scanpaths',return_hist_data=True)
	N = len(data)
	xmin = 0.0001
	alpha = alpha_MLE(data,xmin)
	print "Alpha MLE calculated: " , alpha
	alpha_error = (alpha-1)/(np.sqrt(N))
	print "Alpha error" , alpha_error
	power_law = power_law_pdf(data, alpha, xmin)
	gaussian, mu, variance = gaussian_probabilities(data, xmin=xmin)
	p, ratio = log_likelihood_test(power_law, gaussian)
	print "Likelihood ratio: ", ratio
	print "P-value: " , p

	power_law_samples = sample_power_law_test(N, alpha)
	gaussian_samples = sample_gaussian(N, mu, np.sqrt(variance))
	lognormal_samples = sample_lognormal_mine(N, mu,np.sqrt(variance))
	print "lognormal samples tests"
	print N
	print mu
	print variance
	print lognormal_samples
	l = MLE_lambda(data)
	exp_samples = sample_exponential_mine(N, l)
	exps = exponential_pdf(data, l)
	print "exps"
	print exps
	gaussian, mu, variance = gaussian_probabilities(data, xmin=None)
	p, ratio = log_likelihood_test(gaussian, exps)
	print "Log likelihood test for gaussian vs exponential"
	print "Likelihood ratio: ", ratio
	print "P-value", p

	#and plot on log log plot
	#plt.figure()
	#n,bins, _ = plt.hist(data)
	print " data histogram frequencies"
	print n
	plt.show()
	plaw_n, plaw_bins, _ = plt.hist(power_law_samples,bins=bins)
	print "power law histogram frequencies"
	print plaw_n
	plt.show()
	gauss_n, gauss_bins, _ = plt.hist(gaussian_samples,bins=bins)
	print "gaussian histogram frquencies"
	print gauss_n
	plt.show()
	lognorm_n, lognorm_bins, _ = plt.hist(lognormal_samples, bins=bins)
	print "log normal histogram frequencies"
	print lognorm_n
	plt.show()
	exp_n, exp_bins,_ = plt.hist(exp_samples, bins=bins)
	print "exponential histogram frequencies"
	print exp_n
	plt.show()
	bins = bins[0:len(bins)-1]
	plt.figure()
	plt.loglog(bins,n, label='Data frequencies')
	plt.loglog(bins, plaw_n, label='Power law frequencies')
	plt.loglog(bins, gauss_n,label='Gaussian frequencies')
	plt.loglog(bins, lognorm_n, label='Log normal frequencies')
	plt.loglog(bins, exp_n, label='Exponential frequencies')
	plt.legend()
	plt.show()

	# I mean yeah, it quite significantly here heavier tails 
	# then it should but idk really
	# right, so that works significantly better. Now perhaps I need to try alternate results

	# well, that gives an exceptionally strong gaussian result, like absurdly so
	# probably because hte probability ofthe power law is so shockingly tiny
	# so that's interesting at least
	# now I guess I should turn to some other distributions
	# or at least show the fit distribution of gaussian vs powerlaw
	# check this by plotting, I think



