# okay, this is the basic thing with richard about the seals and their calls finding others with somewhat rapidity
# in amid the cacophany of other's calls. I don't know how it works or how they should find it
#their child. basically richard argues that the seal calls being vocal imitators to some extent
# is so that their parents can find them faster in a mass of corresponding calls
# so that can see what is happening, but I don't know - i.e. the seal parents can follow the gradient
# of their call. first things first is finding an algorithm for the gradient
# so I honestly don't know how that will work
# first we turn a numpy array into it and then iterate through it. it might be slow but could be cool

from __future__ import division
import numpy as np

def euclidean_distance(center, point):
	if len(center)!=len(point):
		raise ValueError('Point and center must have same dimensionality')
	total = 0
	for i in xrange(len(center)):
		total += np.sqaure(center[i] - point[i])
	return np.sqrt(total)

def create_random_colour_matrix(height, width):
	mat = np.zeros((height, width))
	for i in xrange(height):
		for j in xrange(width):
			mat[i][j][0] = np.random.uniform(low=0, high=1) * 255.
			mat[i][j][1] = np.random.uniform(low=0, high=1) * 255.
			mat[i][j][2] = np.random.uniform(low=0, high=1) * 255.
	return mat

def average_point(mat,center,px_radius, image_height, image_width):
	x,y = center
	green_total = 0
	red_total= 0
	blue_total = 0
	number = 0
	for i in xrange(px_radius*2):
		for j in xrange(px_radius*2):
			#check it falls within bounds, then check euclidena distance
		if (x - px_radius) + i <0 or (x-px_radius) + i > image_height:
			if (y-px_radius) + j <0 or (y-px_radius) + j >image_width:
				if euclidean_distance(center, (i,j) <=px_radius):
					green_total+= mat[i][j][0]
					red_total+= mat[i][j][1]
					blue_total+=mat[i][j][2]
					number+=1

	return (green_total/number, red_total/number, blue_total/number)

def matrix_average_step(mat, average_radius, copy=True):
	if len(mat.shape)!=3 or mat.shape[2]!=3:
		raise ValueError('Matrix must be 2d colour image with 3 channels in format h,w,ch')

	height,width, channels = mat.shape
	if not copy:
		new_mat = mat
	if copy:
		new_mat = np.copy(mat)
	#copy so don't mutate on each run through - I can change this behaviour later if I want
	for i in xrange(height):
		for j in xrange(width):
			new_mat[i][j] = average_point(mat, (i,j), average_radius, height,width)
	return new_mat


