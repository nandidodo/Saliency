# okay, this is just for generic utils. such as saving functionality

import numpy as np
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle


#pickle loading and saving functoinality

def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def show_colour_splits(img, show_original = True):
	#assumes img is 4d so we can split along the colour	
	if show_original:
		print "ORIGINAL:"
		plt.imshow(img)
		plt.show()
	print "RED:"
	plt.imshow(img[:,:,0])
	plt.show()
	print "GREEN:"
	plt.imshow(img[:,:,1])
	plt.show()
	print "BLUE:"
	plt.imshow(img[:,:,2])
	plt.show()


def index_distance(indices1, indices2):
	#this finds the euclidian distance. we probably don't realy want any other type tbh
	assert len(indices1) == len(indices2),'indices must have same dimension'
	total = 0
	for i in xrange(len(indices1)):
		total += (indices1[i] - indices2[i]) **2
	return np.sqrt(total)

def max_index_in_array(arr):
	#only works for 2d arrays atm
	maxval = 0
	shape = arr.shape
	indices = [0,0]

	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			if arr[i][j]>maxval:
				maxval = arr[i][j]
				indices=[i,j]
	
	return indices

def get_amplitude_spectrum(img, mult = 255, img_type = 'uint8', show = False, type_convert=True):
	# first we get the fft of the image
	img_amp = np.fft.fft2(img)
	#then we turn it to the amplitude spectrum
	img_amp = np.fft.fftshift(np.abs(img_amp))
	#we ten take logarithms
	img_amp = np.log(img_amp + 1e-8)
	#we resscale to -1:+1 for displays
	img_amp = (((img_amp - np.min(img_amp))*2)/np.ptp(img_amp)) -1
	#we then multiply it out and cast it to type displayable in matplotlib
	if type_convert:
		img_amp = (img_amp * mult).astype(img_type)

	else:
		img_amp = img_amp * mult

	#we then show if we want to
	if show:
		plt.imshow(img_amp)
		plt.show()

	#and then return
	return img_amp

def get_fft(img):
	return np.fft.fft2(img)

def get_magnitude_spectrum(img, show=False, type_convert=True, img_type='uint8'):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	#print magnitude_spectrum
	if type_convert:
		magnitude_spectrum = magnitude_spectrum.astype(img_type)
	
	if show:
		#we plot the original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Original Image')
		plt.xticks([])
		plt.yticks([])
	
	#transformed image
		plt.subplot(131)
		plt.imshow(magnitude_spectrum, cmap='gray')
		plt.title('Magnitude Spectrum')
		plt.xticks([])
		plt.yticks([])
		plt.show()

	#we then return the magnitude spectrum
	return magnitude_spectrum

def get_fft_shift(img):
	f = np.fft.fft2(img)
	return np.fft.fftshift(f)
		


def high_pass_filter(img, filter_width = 10, show = False):

	fshift = get_fft_shift(img)

	rows, cols, channels = img.shape
	crow, ccol = rows/2, cols/2
	#we remove low pass filters by simply dumping a masking window of 60 pixels width across the miage, fshift is the functiondefined to do tht
	fshift[crow-filter_width: crow+filter_width, ccol-filter_width: ccol+filter_width] = 0
	#we start to transform it back
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)

	if show:
		#get original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(img_back, cmap='gray')
		plt.title('Image after HPF')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()

	return img_back



def mean_map(err_map1, err_map2):
	# this really gets the mean
	#asserts
	shape = err_map1.shape
	assert shape == err_map2.shape, 'Error maps are not compatible'
	avg_map = np.zeros(shape)
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			avg_map[i][j] = (err_map1[i][j] + err_map2[i][j])/2.

	return avg_map
	
	
	
