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


#returns the probabiliy of a point given the power law
def power_law(x, alpha,xmin):
	norm = (a-1)/xmin
	power = (x/xmin)**(-1*alpha)
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
	return 1 + (N * (1/MLE_SUM))


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


def find_xmin(data, min_xmin=0, max_xmin=None)
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

def likelihood_ratio(dist1, dist2):
	# here the distributions are assumed to simply be 
	#approximated by vectors of probabilities
	assert len(dist1)==len(dist2):
	ratio = 0
	for i in range(len(dist1)):
		ratio += np.log(dist1[i] - dist2[i])
	return ratio
	# in log space it's a simple sun of subtraction isntead of a product of divisions

def likelihood_variance(dist1, dist2):
	N = len(dist1)
	assert N ==len(dist2), 'distributions must have same length'
	mu1 = np.mean(dist1)
	mu2 = np.mean(dist2)
	total = 0
	for i in range(N):
		total += ((dist1[i]-mu1)-(dist2[i]-mu2)**2)
	return total/N

# I need to begin being able to calculate log likelyhood ratios

def normalised_log_likelihood_ratio(ratio,N,std):
	assert ratio>=0,'Ratio must be positive'
	if ratio=0:
		ratio=-10*100
	ratio = np.log(ratio)
	return np.sqrt(N) * ratio * std


#I've got power law statistica tests
# now I just need todothe same to fit to log normal and exponential and others
# and normal obviosuly and measure degree of fit to be reasnoable!
