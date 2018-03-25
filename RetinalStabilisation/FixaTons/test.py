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


#okay, that is really really REALLY COOL! it works. That's amazing.
#The fixatons people did a really good job with this!