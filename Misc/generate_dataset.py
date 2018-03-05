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
		save_array(filelist, save_dir)
	return filelist

def generate_dataset(data_fname, samples_per_image=50, viewport_width=100, viewport_height=100, save_name=None):
	data = load_array(data_fname)
	pan_dims = data.shape
	threeD=False
	twoD = False
	if pan_dims.length==3:
		twoD=True
	if pan_dims.length==4:
		threeD=True
	if pan_dims<3 or pan_dims >4:
		raise AssertionError('Dimensions of panorama image must be two or three dimensional (plus number of images')
	
	if twoD:
		h,w = pan_dims
	if threeD:
		h,w,channels = pan_dims
	assert samples_per_image>0,'Must have 1 or more sample per image'
	assert viewport_width>0 and viewport_width<=w, 'Viewport must be smaller than whole panorama image'
	assert viewport_height>0 and viewport_height<=h, 'Viewport must be smaller than whole panorama image'

	if save_name is not None:
		assert type(save_name)==type(' ') and len(save_name)>0, 'Save name must be a string of length at least 1'
	
	#now asserts and setup is done, begin the main loop
	#just assume 2d for now!
	generated_data = []
	for i in xrange(len(data)):
		pan_img = data[i]
		#reshape
		pan_img = np.reshape(pan_img, (h,w))
		#now create samples
		for j in xrange(samples_per_image):
			#so it can never be needing padding in the first place
			#probably good for training
			centre_width = int(np.random.uniform(low=viewport_width, high=w-viewport_width))
			centre_height = int(np.random.uniform(low=viewport_height, high=h-viewport_height))
			#get the viewport
			viewport = move_viewport(pan_img, (centre_width, centre_height), viewport_width, viewport_height)
			#add it to data
			generated_data.append(viewport)
	
	#return the data
	generated_data = np.array(generated_data)
	if save_name is not None:
		save_array(generated_data, save_name)
	return generated_data

if __name__ == '__main__':
	#collect_files_and_images(rootdir='../datasets/Benchmark/BenchmarkIMAGES/', save_dir='benchmarkData')
	#load image
	test = load_array('benchmarkData')
	print test.shape
	#this is slow but it works. I'm not sure how to stream data in to python
	#without requiring it all be in memory!
	
	
