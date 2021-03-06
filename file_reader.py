
import numpy as np
import scipy
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import cPickle as pickle
#import cv2
import cPickle as pickle
from utils import *
import os


dirname = 'BenchmarkIMAGES/'
crop_size = (1024,1024)
mode = 'RGB'
num_images = 300 

save_dir=''
save_name = 'MIT300_pickle'
default_size = (1024, 1024)
N_splits = num_images
	

def collect_images(dirname,num_images,crop_size=None, mode='RGB'):
	imglist = []
	if crop_size is not None:
		for i in xrange(num_images):
			fname = dirname + 'i'+str((i+1))+'.jpg'
			img = imresize(imread(fname, mode=mode), crop_size)
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


def collect_files_and_images(rootdir, crop_size = default_size, mode='RGB', save = True, save_dir = None):
	filelist = []
	print "IN FUCNTION"
	for subdir, dirs, files in os.walk(rootdir):
		
		print subdir
		print dirs
		for file in files:
			print file
			fname = os.fsdecode(file)
			if filename.endswith(".jpg"):
				#if it's jpg then its an image, so we're sorced
				if crop_size is not None:
					img = imresize(imread(filename, mode=mode), crop_size)
				if crop_size is None:
					img = imread(filename, mode=mode)
				filelist.append(img)

	filelist = np.array(filelist)
	if save and save_dir is not None:
		# we save
		save_images(filelist, save_dir, save_name = "_data")
	return filelist

def print_dirs_files(rootdir):
	for subdir, dirs, files in os.walk(rootdir):
		print subdir
		print "  "
		print dirs
		print "  "
		print files
		print "  "


def save_images_per_directory(rootdir, crop_size = default_size, mode='RGB', save=True, save_dir='./', make_dir_name = None):
	#first we check if we want to make a new dir, 
	if make_dir_name is not None:
		assert type(make_dir_name) == str, 'make directory name must be a string'
		save_dir = save_dir +  make_dir_name + '/'
		if not os.path.exists(make_dir_name):
			try:
				os.makedirs(make_dir_name)
			except OSError as e:
				if e.errno!= errno.EEXIST:
					print "error found: " + str(e)
					raise
				else:
					print "directory probably already exists despite check"
					raise

	# we walk the filepath
	#subdirs, dirs, files = os.walk(rootdir)
	print os.walk(rootdir)
	if not save:
		total_list = []
	for subdir, dirs, files in os.walk(rootdir):
		filelist = []
		
		for file in files:
			#check it's actually a jpg incase we have random junk in there
			#print file
			#print dirs
			#print subdir
			filename = os.path.basename(file)
			if file.endswith(".jpg") or file.endswith(".jpeg"):
				if crop_size is not None:
					print "IN IMAGE LOOP"
					#print subdir
					print filename
					img = imresize(imread(subdir + '/' + filename, mode=mode), crop_size)
				if crop_size is None:
					img = imread(filename, mode=mode)
				filelist.append(img)
		# now we get the name # split on slash
		
		splits = subdir.split("/")
		#get the last split
		name = splits[-1]
		#this is just a dire hack, but it mightwork
		if name=="":
			name = splits[0]
		name += "_images"
		#check if we have an output file
		if len(dirs) ==0:
			name = splits[-2]
			name = name + "_output"
		#turn filelist into array
		filelist = np.array(filelist)
		# and then save 
		if save:
			print name
			print filelist.shape
			save_array(filelist,save_dir + name)
			print "SAVED: " + name
			print save_dir+name
		# if not save we return all our lists
		if not save:
			total_list.append(filelist)
			print "PROCESSED: " + name
	if not save:
		total_list = np.array(total_list)
		return total_list

	#iterate through subdirs
	#for subdir in subdirs:
		#print subdir


def combine_arrays_into_one(rootdir, save=True, add_name="combined", make_dir_name='', verbose=True):
	if make_dir_name != '':
		if not os.path.exists(rootdir + make_dir_name):
			try:
				os.makedirs(rootdir + make_dir_name)
			except OSError as e:
				if e.errno!= errno.EEXIST:
					print "error found: " + str(e)
					raise
				else:
					print "directory probably already exists despite check"
					raise
		
	#we now walk the directory
	for subdir, dirs, files in os.walk(rootdir):
		img_arr = []
		out_arr = []
		for file in files:
			filename = os.path.basename(file)
			if '.' not in filename:	#this checks for all files without an extension, which should be our saved files
				if verbose:
					print "combined: " + str(rootdir+'/'+filename)
				arr = load(rootdir + '/' + filename)
				img_type = filename.split('_')[-1] # get the last bit
				if img_type == 'images':
					img_arr.append(arr)
					if verbose:
						print "added to images"
				if img_type == 'output':
					out_arr.append(arr)
					if verbose:
						print "added to outputs"
		#concatenate list of arrays into single mega array
		img_arr = np.concatenate(img_arr)
		out_arr = np.concatenate(out_arr)
		if verbose:
			print "images shape" + str(img_arr.shape)
			print "outputs shape" + str(out_arr.shape)
		if save:
			if make_dir_name == '':
				save_array(img_arr, rootdir + 'images_'+ add_name)
				save_array(out_arr, rootdir + 'ouputs_'+ add_name)
			
			if make_dir_name !='':
				save_array(img_arr, rootdir + make_dir_name + '/' + 'images_' + add_name)
				save_array(out_arr, rootdir + make_dir_name + '/' + 'outputs_' + add_name)
		return img_arr, out_arr



#print "doing file-reader"
#dirname = 'BenchmarkIMAGES/'	
#save_images_per_directory(dirname, save=True, crop_size=(32, 32))
#
def read_image(num, dirnme = dirname, mode='RGB'):
	fname = dirname + 'i' + str(num) +'.jpg'
	return imread(fname, mode=mode)
	
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

def split_img_on_colour(img, mode='RGB'):
	
	if mode== 'RGB':
		red = img[0,:,:]
		blue = img[1,:,:]
		green = img[2,:,:]
		return [red, blue, green]



if __name__ == '__main__':
	#set up our default variables
	print "File Reader Activated"
	
	#I think the max size is 1024x784

	dirname = 'BenchmarkIMAGES/'
	crop_size = (1024,1024)
	mode = 'RGB'
	num_images = 300 

	save_dir=''
	save_name = 'MIT300_pickle'
	default_size = (1024, 1024)
	N_splits = num_images
	

			

# okay, let's try thi ssimply
#fname = './BenchmarkIMAGES/i1.jpg'

#img = imread(fname, mode='RGB')
#print type(img)
#print img.shape

#plt.imshow(img)
#plt.show()

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


