# okay, the aim here is to quickly try to write the cone function - see if that would be
# at all useful or work at all, as a quick feasibility test!

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy
from math import atan2, degrees


# so I need some funtions to convert visual angles and pixels and vice versa, which
# I'm just looking up how to do on the internet - so who knows

def px_to_degrees(stim_width, stim_height, monitor_height, distance, resolution):
	deg_per_px = degrees(atan2(0.5*monitor_height,distance) / 0.5*resolution)
	new_width = deg_per_px*stim_width
	new_height = deg_per_px*stim_height
	return new_width, new_height

def degrees_to_px(stim_width, stim_height, monitor_height, distance, resolution):
	deg_per_px - degrees(atan2(0.5*monitor_height, distance)/ 0.5*resolution)
	new_width = stim_width / deg_per_px
	new_height  = stim_height / deg_per_px
	return new_width, new_height

def euclidean_distance_numpy(point, center):
	if len(point) !=len(center):
		raise ValueError('Point and center must have the same dimensions!')
	return np.sqrt(np.sum(np.square(point - center)))

def euclidean_distance(point, center):
	if len(point) !=len(center):
		raise ValueError('Point and center must have same dimensions')
	total = 0
	for i in range(len(point)):
		total +=  (point[i] - center[i])**2
	total = np.sqrt(total)
	return total

def convert_to_grayscale(img):
	if len(img.shape)!=3:
		raise ValueError('Input image must be three dimensional')
	return img[:,:,1:2] = 0

def foveal_colour(input_img, foveal_radius):
	if len(input_img.shape)!=3:
		raise ValueError('Input image must be three dimensional - i.e. colour image')
	if foveal_radius<=0:
		raise ValueError('Foveal radius should be positive and greater than 0')

	#copy image since need a reference to old
	img = np.copy(input_img)
	#convert to grayscale
	img  = convert_to_grayscale(img)
	height, width, channels = img.shape
	center = (height//2, width//2)
	fovea_radius = 100
	#add back in the colour section
	for i in xrange(height):
		for j in xrange(width):
			#just iterate over the image
			dist = euclidean_distance((i,j), center)
			if dist <= fovea_radius:
				img[i][j][1:2] = input_img[i][j][1:2]
	return img

def foveal_colour_with_random_cones(input_img, foveal_radius, cone_radius, num_cones):
	if cone_radius<=0:
		raise ValueError('Cone radius must be a positive number greater than 0')
	if num_cones<=0:
		raise ValueError('Number of cones must be a positive number greater than 0')
	#add colour to the img


	img = foveal_colour(input_img, foveal_radius)
	height, width, channels = img.shape

	#distribute the cones randomly through the image
	cones_added = 0
	while cones_added<=num_cones:
		#generate random x and y
		nh = int(np.random.uniform(low=0, high=1)*height)
		nw = int(np.random.uniform(low=0, high=1)*width)
		#check not already part of a cone
		if img[nh][nw][1]!=0:
			# i.e. it's not already part of a cone
			# then add the cone		




# okay, now that's done I need to test the image that it works okay

if __name__ == '__main__':
	data = np.load('benchmarkData.npy')
	old_img = data[0]
	plt.imshow(old_img)
	plt.show()
	img = np.copy(old_img)
	# okay, try grayscaling
	img[:,:,1:2]=0
	print img.shape
	plt.imshow(img)
	plt.show()
	height, width, channels = img.shape
	center = (height//2, width//2)
	fovea_radius = 100
	#add back in the colour section
	for i in xrange(height):
		for j in xrange(width):
			#just iterate over the image
			dist = euclidean_distance((i,j), center)
			if dist <= fovea_radius:
				img[i][j][1] = img[i][j][0]
				img[i][j][2] = img[i][j][0]
				#img[i][j][:] = 0 # blackout the image to test
				# it works... yay!



	plt.imshow(img)
	plt.show()
	# so this is actually pretty simple and nice. now let's try a random cone distribution



