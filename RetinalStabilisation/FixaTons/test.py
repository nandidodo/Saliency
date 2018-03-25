import FixaTons as FT

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


datasets = FT.list.datasets()
#create a stimuli object for this
stimuli = []
for dataset in datasets:
	stims = FT.list.stimuli(dataset)
	for stim in stims:
		stimuli.append((stim, dataset))

print stimuli

# so this is fairly straightforward. Now let's get all scanpaths in a reasonable fashion!

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