# okay, let's try a log polar sampling. This should be really fun. I'm really not sure how to recreate the image in a way we can display it though. that's realy bad... ugh!
# this does seem to be extremely important as it's how the brain actually perceives these thigns
# and writing it up seems like a very valuable thing to be doing,so we can understnad just how it works
# and also understand just how julia works also, as a language, as writing the algorithm in that first, although in python also seems to be very important


## let's try this as a very very simple loop. it'll be death slow in python, but that won'tbe too horrendous

# I wonder actually let's define a really simple and hacky transform which just blitzes the resolutoin of the thing out so it makes sense, with the resolution falling off at each time. I honestly don't know. also, we need to know what to do if we reach edges and so forth, but I don't know... argh. dagnabbit edge effects. let's try out a cool image transform and see what happens this is just artificially decreasing the resolution to test and see if anything interesting is happening

# ythis is just a file where I try to figure out some simple python foveal image transforms to see what's up here, because this will undoubtedly be very useful and cool. ew can do this... let's go!

import numpy as np
import matplotlib.pyplot as plt
from utils import *

# let's just import an image
imgs = load_array('testimages_combined')
print imgs.shape
img = imgs[40]

# okay, I forgot about multiple channels. let's just grayscale this
img = img[:,:,0]
img = np.reshape(img, (img.shape[0], img.shape[1]))
plt.imshow(img)
plt.show()

# okay great. I forgot they were only doing stuff on the 100/100 iamges. oh well. we'll have to fix that later, but for now it doesn't matter. let's first just figure out a simple manual for loop thing that decreases the resolution at each tick of the image from the centre

#not sure to the extent to which the resolution should decrease. I really have no clue. guess I should find out, but let's test it for a bit

# this is going to be horrendously inefficient, as it is python, and not vectorised code

#okay, our very simple and currently primitive distance function

# okay, well, thisdoesn' really work, but nevertheless it's cool. I should also go and have a shower now, I think
def resolution_by_distance(dist):
	if dist<=20:
		return 0
	#if dist>20:
	#	return int(dist/30)
	if dist > 20 and dist <40:
		return 1
	if dist > 40:
		return 2


def manual_resolution_decrease(img, centre):
	shape = img.shape
	lenx = shape[0]
	leny = shape[1]
	centrex = centre[0]
	centrey = centre[1]

	#init our new img
	newimg = np.zeros(shape)

	#do asserts
	assert centrex > 0 and centrex <lenx, 'x coordinate of centre outside image'
	assert centrey > 0 and centrey <leny, 'x coordinate of centre outside image'

	for i in xrange(lenx):
		for j in xrange(leny):
			# we calculate distance from centre
			dist = np.sqrt((i-centrex)**2 + (j-centrey)**2)
			# we have soem distance function which maps to numbers of boxes to average over?
			res = resolution_by_distance(dist)
			print res
			#we initialise total and num with current position
			total = img[i][j]
			num = 1
			if res > 0:
				for k in xrange(res):
					for l in xrange(res):
						propx = i-k
						propy=j-l
						#if it's within boundaries
						if propx >-1 and propx <lenx and propy>-1 and propy<leny:
							total += img[propx][propy]
							num +=1
						propx = i+k
						propy=j+l
						#if it's within boundaries
						if propx >-1 and propx <lenx and propy>-1 and propy<leny:
							total += img[propx][propy]
							num +=1
			avg = total/num
			newimg[i][j]=avg
	return newimg


centre = (50, 50)
fovimg = manual_resolution_decrease(img, centre)
plt.imshow(fovimg)
plt.show()


