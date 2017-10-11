# okay, this script is just meant to help us read in the files to the requisite numpy arrays which we can then pickle and save. it seems fairly reasonable, but I realy don't know how to do it. If I was sensible I would use classes, but I'm not, so I'm going to fail terrible instead!

import numpy as np
import scipy
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import cPickle as pickle
#import cv2
import cPickle as pickle
from utils import *

#I think the max size is 1024x784

dirname = './BenchmarkIMAGES/'
crop_size = (1024,1024)
mode = 'RGB'
num_images = 300

save_dir=''
save_name = 'MIT300_pickle'
N_splits = num_images

def collect_images(dirname,num_images,crop_size=None, mode='RGB'):
	imglist = []
	if crop_size is not None:
		for i in xrange(num_images):
			fname = dirname + 'i'+(i+1)+'.jpg'
			img = imresize(imread(fname, mode=mode), crop_size))
			imglist.append(img)
	
		#we turn imglist into a flat array and return it
		imglist = np.array(imglist)
	if crop_size is None:
		for i in xrange(num_images):
			fname = dirname + 'i' + (i+1) +'jpg'
			img = imread(fname, mode=mode)
			imglist.append(img)
	
	imglist = np.array(imglist)
	return imglist
	
def save_images(imgarray, save_dir, save_name, N_splits = None):
	if N_splits is None:
		fname = save_dir+save_name
		save(imgarray, fname)
	if N_splits is not None:
		N = len(imgarray)/N_splits
		for i in xrange(N_splits):
			#we do our array slicing
			arr = imgarray[(N*i):(N*(i+1)), :,:,:]
			fname = save_dir + save_name +'_'+i
			save(arr, fname)


			

# okay, let's try thi ssimply
#fname = './BenchmarkIMAGES/i1.jpg'

#img = imread(fname, mode='RGB')
#print type(img)
#print img.shape

#plt.imshow(img)
#plt.show()

# let's try a really hacky resize method to see if it works - I mean it kind of does. it looks kind of ugly, but oh well, and these images are going to be big. we might need to tone this down in production to get it to a reasonable size, nevertheless, we can do this realtively simply. we just have one more thing to test before we have this script being workable (well, two, including the pickling!)
#imgarray = []
 
#resized = imresize(img, (1024,1024))
#plt.imshow(resized)
#plt.show()

#img2 = imread(fname,mode='RGB')
#resized2=imresize(img2, (1024, 1024))

#resized.append(resized2)
#print resized.shape

#imgarray.append(resized)
#imgarray.append(resized2)
#print imgarray.shape
#arr = np.array(imgarray)
#print arr.shape
# okay, yes! that wokrs. now let's make a proper script to do this


#print resized.shape



