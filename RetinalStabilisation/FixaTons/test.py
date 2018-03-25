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

if __name__ =='__main__':
	investigate_scanpath_lengths('fixaton_scanpaths')
