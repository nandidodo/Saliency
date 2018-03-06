# okay, so the aim of this is to generate the dataset from the other image data I have
#it just samples each one with the viewport and uses that in the dataset
#it should probably do 3d also except the panorama is only really okay for 2d images
# also I'm going to ahve to talk to julia in a bit... dagnabbit!



import numpy as np
from panorama import *
from utils import *
import os
import scipy
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt

def collect_files_and_images(rootdir, crop_size = None, mode='RGB', save = True, save_dir = None):
	filelist = []
	print "IN FUCNTION"
	for subdir, dirs, files in os.walk(rootdir):
		# this will save them all in one enormous file, which makes sense, but is dire
		#print subdir
		#print dirs
		#print subdir
		#print files
		for file in files:
			print file
			filename = os.path.basename(file)
			if filename.endswith(".jpg"):
				#if it's jpg then its an image, so we're sorced
				if crop_size is not None:
					img = imresize(imread(rootdir+ filename, mode=mode), crop_size)
				if crop_size is None:
					print filename
					img = imread(rootdir + filename, mode=mode)
				print 'file found'
				filelist.append(img)

	filelist = np.array(filelist)
	if save and save_dir is not None:
		# we save
		np.save(save_dir, filelist)
	return filelist

def generate_dataset(data_fname, samples_per_image=50, viewport_width=100, viewport_height=100, save_name=None):
	data = np.load(data_fname)
	assert samples_per_image>0,'Must have 1 or more sample per image'

	if save_name is not None:
		assert type(save_name)==type(' ') and len(save_name)>0, 'Save name must be a string of length at least 1'
	
	#now asserts and setup is done, begin the main loop
	#just assume 2d for now!
	generated_data = []
	for i in xrange(len(data)):
		#cut down to 2d
		pan_img = data[i]
		pan_img = pan_img[:,:,0]
		h,w = pan_img.shape
		print pan_img.shape
		assert viewport_width>0 and viewport_width<=w, 'Viewport must be smaller than whole panorama image'
		assert viewport_height>0 and viewport_height<=h, 'Viewport must be smaller than whole panorama image'
		print "Loading image: " + str(i)

		#reshape
		pan_img = np.reshape(pan_img, (h,w))
		#now create samples
		for j in xrange(samples_per_image):
			print "Sample: " + str(j)
			#so it can never be needing padding in the first place
			#probably good for training
			centre_width = int(np.random.uniform(low=viewport_width, high=w-viewport_width))
			centre_height = int(np.random.uniform(low=viewport_height, high=h-viewport_height))
			#get the viewport
			viewport = move_viewport(pan_img, (centre_height, centre_width), viewport_width, viewport_height)
			#add it to data
			generated_data.append(viewport)
	
	#return the data
	generated_data = np.array(generated_data)
	if save_name is not None:
		np.save(save_name, generated_data)
	return generated_data

if __name__ == '__main__':
	#collect_files_and_images(rootdir='../datasets/Benchmark/BenchmarkIMAGES/', save_dir='benchmarkData')
	#load image
	#test = np.load('benchmarkData.npy')
	#print test.shape
	#generate_dataset('benchmarkData.npy', save_name='panoramaBenchmarkDataset')
	test = np.load('panoramaBenchmarkDataset.npy')
	print test.shape
	#perfect!
	##for i in xrange(10):
	#	plt.imshow(test[i])
	#	plt.show()

	#this is slow but it works. I'm not sure how to stream data in to python
	#without requiring it all be in memory!
	#okay, so this can be loaded in instantly. that's amazing yay!
	
	
