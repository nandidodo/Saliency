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
	img[:,:,1:2] = 0
	return img

def convert_to_grayscale_mean(img):
	if len(img.shape)!=3:
		raise ValueError('Input image must be three dimensional')
	return np.mean(img, -1)

def convert_to_grayscale_adjust(img):
	if len(img.shape)!=3:
		raise ValueError('Input image must be three dimensional')
	#not sure wherethese magic numbers come from, just some person on stack overflow?
	r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
	gray = 0.2989*r + 0.5870*g + 0.1140*b
	return gray

def greyscale_expand(img):
	if len(img.shape!=2):
		raise ValueError('The initial image for this function must be two dimensional')
	h,w = img.shape
	exp = np.zeros((h,w,3))
	exp[:,:,0] = img
	return exp

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
	for i in xrange(fovea_radius*2):
		for j in xrange(fovea_radius*2):
			#just iterate over the image
			x = (center[0]-fovea_radius)+i
			y = (center[1] - fovea_radius)+j
			dist = euclidean_distance((x,y), center)
			if dist <= fovea_radius:
				img[x][y][1:2] = input_img[x][y][1:2]
	return img

def foveal_colour_with_random_cones(input_img, foveal_radius, cone_radius, num_cones):
	if cone_radius<=0:
		raise ValueError('Cone radius must be a positive number greater than 0')
	if num_cones<=0:
		raise ValueError('Number of cones must be a positive number greater than 0')
	#add colour to the img
	if len(input_img.shape)!=3:
		raise ValueError('Input image must be three dimensional')
	h,w, ch = input_img.shape
	if fovea_radius>=h or fovea_radius>=w:
		raise ValueError('Foveal radius cannot be greater than the image size')
	if cone_radius>=h or cone_radius>=w:
		raise ValueError('Cone radius cannot be greater than the image size')



	img = foveal_colour(input_img, foveal_radius)
	height, width, channels = img.shape

	#distribute the cones randomly through the image
	cones_added = 0
	while cones_added<=num_cones:
		#print "in cone adding loop"
		#generate random x and y
		nh = int(np.random.uniform(low=0, high=1)*(height-cone_radius))
		nw = int(np.random.uniform(low=0, high=1)*(width-cone_radius)) # ensure does not end up outside of the image
		#check not already part of a cone
		if img[nh][nw][1]==0:
			# instead of scanning the whole image, just scan the part around the 
			#actual cone to check instead, would save a lot!
			cones_added+=1
			print "cone added"
			for i in xrange(cone_radius*2):
				for j in xrange(cone_radius*2):
					x = (nh-cone_radius)+i
					y = (nw-cone_radius)+j
					#print "Loop location:  "+ str(x)+ " " + str(y)
					#print "Center: " + str(nh) + " "+ str(nw)
					dist = euclidean_distance((x,y),(nh,nw))
					#print dist
					if dist <=cone_radius:
						#print "changing image part"
						img[x][y][1:2] = input_img[x][y][1:2]
			# i.e. it's not already part of a cone
			# then add the cones
	return img





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
	#for i in xrange(height):
	#	for j in xrange(width):
	#		#just iterate over the image
	#		dist = euclidean_distance((i,j), center)
	#		if dist <= fovea_radius:
	#			img[i][j][1] = img[i][j][0]
	#			img[i][j][2] = img[i][j][0]
				#img[i][j][:] = 0 # blackout the image to test
				# it works... yay!
	imf = foveal_colour_with_random_cones(old_img, 100, 10, 40)
	plt.imshow(imf)
	plt.show()




	#plt.imshow(img)
	#plt.show()
	# so this is actually pretty simple and nice. now let's try a random cone distribution



