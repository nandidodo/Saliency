# okay, this is just meant to be a very simple file for plotting and stuff which is easy to use
# and let's us get our datasets fairly standardly
# for the plots used in the paper, so not difficult.
# we'll obviously comment each method with how it is used when we push this to github properly
# orwhatever to sort it out

import matplotlib.pyplot as plt
from utils import *
from experiments import *
from tests import *

#first we load the file so it makes sense
img_fname = 'testimages_combined'

imgs = load_array(img_fname)
print imgs.shape


def show_img_red_green(img_fname, indices):
	#get imgs
	imgs = load_array(img_fname)
	#get our images we want to show
	imgs_to_show = []
	N = len(indices)
	for i in xrange(N):
		imgs_to_show.append(imgs[indices[i]])
	reds = []
	greens = []
	for j in xrange(N):
		img = imgs_to_show[i]
		reds.append(img[:,:,0])
		greens.append(img[:,:,1])
	#now we begin plotting
	#not sure how to do this. it would be annoying. what even is the name of the dataset. I should at least get this to work... dagnabbit. not enough time at all. I just have had no focus the past few days and it really shows. nothing is done... argh!
	for k in xrange(N):
		#full img
		plt.subplot(k+1, 3,1)
		plt.imshow(imgs_to_show[k])
		plt.title('original image')
		plt.xticks([])
		plt.yticks([])
		#red img
		plt.subplot(k+1, 3,2)
		plt.imshow(reds[k],cmap='gray')
		plt.title('red compoment')
		plt.xticks([])
		plt.yticks([])
		#green image
		plt.subplit(k+1, 3,3)
		plt.imshow(greens[k], cmap='gray')
		plt.title('green component')
		plt.xticks([])
		plt.yticks([])
	plt.show()
		
	
	
		
