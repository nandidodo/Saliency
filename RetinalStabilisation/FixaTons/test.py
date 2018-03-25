
from __future__ import division
import FixaTons as FT
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
# I realy hate python's imports. They are totally rubbish
# in any case let's hope this works

# I wonder how they do it with ths function composition here
#it's itneresting, so I shuold definitely look into the source code

print FT.list.datasets()
print FT.list.stimuli('SIENA12')

DATASET_NAME = 'SIENA12'
STIMULUS_NAME = 'land.jpg'
SUBJECT_ID = 'GT_10022017'
"""
stimulus_matrix = FT.get.stimulus(DATASET_NAME, STIMULUS_NAME)
saliency_map_matrix = FT.get.saliency_map(DATASET_NAME, STIMULUS_NAME)
fixation_map_matrix = FT.get.fixation_map(DATASET_NAME, STIMULUS_NAME)

print 'Stimulus dims = ', stimulus_matrix.shape
print "saliency map dims = ", saliency_map_matrix.shape
print "fixation map matrix = ", fixation_map_matrix.shape

scanpath = FT.get.scanpath(DATASET_NAME, STIMULUS_NAME, subject = SUBJECT_ID)
print scanpath.shape
print scanpath
print "this scanpath has ", len(scanpath), "fixations."

#show the map
FT.show.map(DATASET_NAME, STIMULUS_NAME, showSalMap=True, showFixMap=False, wait_time=5000, plotMaxDim=1024)

# that's cool! all of this works even though presumably it requires the cv2 library
#which I thought Icuoldn't figure out how to install(!)

#show a single scanpath
FT.show.scanpath(DATASET_NAME, STIMULUS_NAME, subject=SUBJECT_ID,
	animation=True, wait_time=0, putLines=True, putNumbers = True,
	plotMaxDim=1024)
"""

#okay, that is really really REALLY COOL! it works. That's amazing.
#The fixatons people did a really good job with this!

# okay, so aim from now on will be to just dump the scanpaths into a database
# or somethingwith a common data  format hopefully
# I don't really know
# I can always save the mas python objects but it will be quite bad
# instead I should probably have the python object just store the metadata
# and then contain a filename -i.e. a pointer to the scanpath npy file
# which would seem reasonable, and Icould even have it as an mmapped file if need be
# although that is unlikely to actually be required

# so this is fairly straightforward. Now let's get all scanpaths in a reasonable fashion!

"""
subjects = FT.list.subjects(DATASET_NAME, STIMULUS_NAME)

# so to get all scanpaths it would then be trivial. let's examine some of them
for subject in subjects:
	scanpath = FT.get.scanpath(DATASET_NAME, STIMULUS_NAME, subject = subject)
	print "Scanpath for subject: ", subject
	print "Scanpath dimension is ", scanpath.shape
	# okay, well that works easily
	# but the trouble is probably more widespread
	# in that all the results are not the same, would need to be stored as a list?
	#and how would I do the statistics. What do the fields actually mean in this case?
	# ah I see, the format is "x-pixel", "y-pixel", "initial-time", "final-time"
	# I'm not sure how I wuold actually be able to figure out the scanpath statistics 
	# in a requisite way
	# although I have a fair amount of data available, which is good!
	"""


def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))


def load(fname):
	return pickle.load(open(fname, 'rb'))


def save_all_scanpaths():
	datasets = FT.list.datasets()
	#create a stimuli object for this
	stimuli = []
	for dataset in datasets:
		stims = FT.list.stimuli(dataset)
		for stim in stims:
			stimuli.append((stim, dataset))

	print stimuli

	#now get it
	scanpaths = []
	for stimulus in stimuli:
		stim, dataset = stimulus
		subjects = FT.list.subjects(dataset, stim)
		for subject in subjects:
			#create  a scanpath object with requisite metadata
			scanpath = {}
			scanpath['dataset'] = dataset
			scanpath['stimulus'] = stimulus
			scanpath['subject_id'] = subject
			scanpath['scanpath'] = FT.get.scanpath(dataset, stim, subject = subject)
			scanpaths.append(scanpath)

	#now I need to save the scanpaths
	print "scanpaths: ", len(scanpaths)
	save(scanpaths, 'fixaton_scanpaths')



# okay, now I can load the scanpaths object I created and do a whole bunch oftests on it
# which is really realy nice
# for statistical purposes, which is great, and hopefully figure uot how to do a levy test
# and having all the data here trivially is really quite perfect
# this data munging was not difficult at all!

# let's look at the scanpath lengths to see if anythin ginteresting is there
def investigate_scanpath_lengths(scanpath_fname,save_name=None):
	scanpaths = load(scanpath_fname)
	lengths = []
	#do a loop
	for scanpath in scanpaths:
		sh = scanpath['scanpath'].shape
		assert sh[1]==4, 'Scanpath data is incomplete'
		lengths.append(sh[0]) # as that's the one
	lengths = np.array(lengths)
	avg = np.mean(lengths)
	var = np.var(lengths)
	print "mean scanpath length: ", avg
	print "variance of scanpath lengths", var
	print "standard deviation of scanpath lengths", np.sqrt(var)

	#save
	if save_name is not None:
		np.save(save_name, lengths)
	#plot the histogram
	plt.hist(lengths)
	plt.title('Histogram of lengths of the scanpaths')
	plt.show()


# okay, so the length of the averate scanpath in here is quite low
# and just generally bad. This might cause some statistical problems
#for me to detect whether it is a levy path or not. Might still be able to do something
# useful however, I really do not know!

# what sort of things wuold be useful for this
# presumably  Iwould like the distribution ot distances between scanpaths
# in euclidean distance, so hopefully that should not be to difficult
# so let's obtain that

def euclidean_distance(p1, p2):
	assert len(p1)==len(p2), 'points must be of same dimension'
	total = 0
	for i in range(len(p1)):
		total += np.square(p1[i] - p2[i])
	return np.sqrt(total)

def euclidean_distance_unsafe(p1, p2):
	return np.sqrt(np.sum(np.square(p1-p2)))


# this ignores the timing info. Not sure how useful that is?
def get_saccade_distances(scanpath_fname, plot=True, save_name=None, info=True, return_hist_data=False):
	scanpaths = load(scanpath_fname)
	distances = []
	for scanpath in scanpaths:
		paths = scanpath['scanpath']
		length = paths.shape[0]
		for i in xrange(length-1):
			path1 = paths[i]
			p1 = path1[0:1]
			path2 = paths[i+1]
			p2 = path2[0:1]
			#append the distance
			dist = euclidean_distance(p1, p2)
			if dist <=10000:
				distances.append(dist)
			#distances.append(euclidean_distance(p1,p2))

	distances = np.array(distances)

	if info:
		length = len(distances)
		print "Number of scanpath distances calculated: ", length
		mean = np.mean(distances)
		print "Mean scanpath distance: " , mean
		variance = np.var(distances)
		print "Variance of scanpath distance: " , variance
		std = np.sqrt(variance)
		print "Standard deviation of scanpath distance: ", std
		max_distance = np.max(distances)
		print "Maximum scanpath distance: ", max_distance
		min_distance = np.min(distances)
		print "Minumum scanpath distance: ", min_distance
	
	#plot if necessary
	if plot:
		#n,bins,patches = plt.hist(distances, bins=15)

		#log_bins = np.logspace(np.log(0.01), np.log(2700),20)
		#print log_bins
		log_bins=20
		n, bins, patches = plt.hist(distances, bins=log_bins)
		plt.title('Distribution of distancs between successive scan paths')
		plt.show()

	if save_name is not None:
		np.save(save_name, distances)

	# I'm not sure if it's a log tailed distibution
	#vast majority are tiny, but seemingly dwarfed by giant one,
	#I'm really not sure!

	if return_hist_data:
		return distances, n,bins,patches

	return distances


def get_fixation_durations(scanpath_fname, plot=True, save_name=None, info=True, return_hist_data=False):
	scanpaths = load(scanpath_fname)
	differences = []
	for scanpath in scanpaths:
		paths = scanpath['scanpath']
		length = paths.shape[0]
		for i in xrange(length):
			path = paths[i]
			differences.append(path[3] - path[2])

	differences = np.array(differences)

	if info:
		length = len(differences)
		print "Number of fixation durations calculated: " , length
		mean = np.mean(differences)
		print "Mean duration of fixation: " , mean
		var = np.var(differences)
		print "Variance of fixation durations: " , var
		std = np.sqrt(var)
		print "Standard deviation of fixation durations: " , std
		max_duration = np.max(differences)
		print "Maximum fixation duration: " , max_duration
		min_duration = np.min(differences)
		print "Minumum fixation duration: " , min_duration

	if plot:
		n, bins, patches = plt.hist(differences)
		plt.title('Distribution of fixation durations')
		plt.show()

	if save_name is not None:
		np.save(save_name, differences)

	if return_hist_data:
		return differences, n,bins,patches

	return differences


def get_scanpath_durations(scanpath_fname, plot=True, save_name=None, info=True):
	scanpaths = load(scanpath_fname)
	differences = []
	for scanpath in scanpaths:
		paths = scanpath['scanpath']
		length = paths.shape[0]
		first = paths[0]
		last = paths[length-1]
		differences.append(last[3] - first[2])

	differences = np.array(differences)

	if info:
		length = len(differences)
		print "Number of scanpath durations calculated: " , length
		mean = np.mean(differences)
		print "Mean duration of scanpaths: " , mean
		var = np.var(differences)
		print "Variance of scanpath durations: " , var
		std = np.sqrt(var)
		print "Standard deviation of scanpath durations: " , std
		max_duration = np.max(differences)
		print "Maximum scanpath duration: " , max_duration
		min_duration = np.min(differences)
		print "Minumum scanpath duration: " , min_duration

	if plot:
		plt.hist(differences)
		plt.title('Distribution of scanpath durations')
		plt.show()

	if save_name is not None:
		np.save(save_name, differences)

	return differences



# a simple way to get it is just to plot it on a log log scale,
# and it shuold be a straight line
# so what wuold it be - it would be log number of points within the bin
# vs log bin size = distance betwene points


# okay, well figuring these things out is all well and good
# but in reality  I need specific tests to check the levy flight hypothesis
# I simply do not know these at the moment, waht I should see
# or what distributions I shold check
# even though I actually have a fair amount of statistics now able to be generated
# about the scanpaths, they still are not that informative to be honest
# so I honestly do not even know what to look for
# this is going to require reading some of the papers in detail
# and trying to understand how they would check
#whether it is actually a levy flight or not!

def get_middle_of_bins(bins):
	middles = []
	for i in xrange(len(bins)-1):
		start = bins[i]
		end = bins[i+1]
		div = (end - start)/2
		middle = start + div
		middles.append(middle)

	middles = np.array(middles)
	return middles

def log(x):
	#our safer log function
	for i in xrange(len(x)):
		if(x[i]==0):
			x[i]=0
		else: 
			x[i] = np.log(x[i])
	return x


def log_log_plot_mine(x,y, xlabel, ylabel, title):
	print "In log log plot"
	print x
	print y
	#x = np.log(np.log(x))
	#y = np.log(np.log(y))
	x =log(x)
	y=log(y)
	print x
	print y
	fig = plt.figure()
	plt.plot(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

	# yeah, I don't know how to do this effectively
	# or how even to plot this
	# I'm still pretty sure it's not log distributed at all
	#dagnabbit!
def log_log_plot(x,y, xlabel, ylabel,title):
	fig = plt.figure()
	plt.loglog(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()




if __name__ =='__main__':
	#investigate_scanpath_lengths('fixaton_scanpaths')

	#distances = get_saccade_distances('fixaton_scanpaths')
	#durations = get_fixation_durations('fixaton_scanpaths')
	#durations = get_scanpath_durations('fixaton_scanpaths')
	distances, n,bins,patches = get_saccade_distances('fixaton_scanpaths',return_hist_data=True)
	#print type(n)
	#print len(n)
	#print type(bins)
	#print len(bins)
	#print type(patches)
	#print len(patches)
	#middles = get_middle_of_bins(bins)
	print bins
	print bins[0:len(bins)-2]
	middles = bins[0:len(bins)-1]
	print type(middles)
	print len(middles)
	log_log_plot_mine(middles, n, 'Log Distance between saccades', 'Log Number of saccades','Log log plot of distance between saccades vs number of saccades')
	
	# yeah, just generally this is very clearly not a log log plot
	# dagnabbit. It's not power law distribted at all, but how is it distribted?
	# I'm not sure how I shuold model it - perhaps as a standard brownian motion?
	# perhaps there just are not long enough scan paths, so I don't nkow?
	# it could be a fun little shot paper to very easily write up
	# what if I had more data?
	# like richard's data or whatever!
	# like realistically 8 is probably not enough for a levy flight to emerge
	# to any degree of significance... dagnabbit!

	# yeah, so far as I can tell, at the moment it doesn't at all follow the log log plot
	# which is quite bad, though I'm almost certainly doing something wrong
	# it's obviously not at all a striahgt line, in factthe decrease is probably
	# greater than the log. dagnabbit. Nevertheless, it's not bad that in a day
	# I've got some results that either confirm nor deny the hypothesis
	# which is not at all bad!
	# I mean I'll need to do some actual tests, but it's not looking good!
	# but that's actually an interesting result in and of itself
	# espeically if I can get more data
	# and beef up the maths and statistics of this
	# then this could legitimately be a useful result to restart a kind of controversty
	# and might even get published somewhere
	# even thoguh it took me just a few days (if that!) to get it sorted
	# which would not be bad at all!
	# what if I don't do it from the histogram but generate it directly from the data
	# but what would that even look like in a serious way? I simply do not know... dagnabbit!
	# I suppose I could see if fiddling with the number of bins would have an actual effect?
	# nope, it's very clear generally that it's still an exponential curve downward
	# so it's a hyper exponential display?
	# basically yeah, this data does quite clearly not follow the log log plot at all!
	# at all. So then what should my response be?
	# lie it is not just a standard normal distribution right?
	# it is just the distribution of distances that I need to look at right?
	# if it's a downward sloping exponential then what does that mean???
	#argh! well that's a bit of a dissapointment there to be honest
	# at least it makesthings slightly more confusing in a lot of ways
	# in how to write it up and so far, as it's not an actual discovery
	# but I eally don't know what to do wrt this then actually
	# guess just put it on the backburner
	# as it's quite clearly not power law distributed
	# but rather exponential decreasing-  perhaps whatever, I don't know
	# could just be a gaussian or whatever. Should talk to richard about it, I don't know
	# perhaps log normal. I guess this is something I'm gong to haev to look up slightly more
	# but whatever it is, it's quite clearly NOT a power law distribution
	#dagnabbit!
	# generally I just don't think that the data is log log distributed at all
	# except there is a weird intermediate sort of zone that makes no sense where it declines
	# probably normally and then later in the tails there is heaviness due to a few huge random ones perhas?