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
		total += (center[i] - point[i])**2
	return np.sqrt(total)

def create_random_colour_matrix(height, width):
	mat = np.zeros((height, width,3))
	for i in xrange(height):
		for j in xrange(width):
			mat[i][j][0] = (np.random.uniform(low=0, high=1) * 255.)
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
			#print "going round loop: " + str(i) + " " + str(j) +" " + str(x) + " " + str(y)
			#print x-px_radius+i 
			#print y-px_radius +j
			xpoint = x - px_radius + i
			ypoint = y - px_radius + j
			#print euclidean_distance(center, (i,j))
			#check it falls within bounds, then check euclidena distance
			if xpoint >=0 and xpoint <= image_height:
				if ypoint >=0 and ypoint + j <=image_width:
					if euclidean_distance(center, (xpoint, ypoint)) <=px_radius:
						#print "adding to average"
						green_total+= mat[i][j][0]
						red_total+= mat[i][j][1]
						blue_total+=mat[i][j][2]
						number+=1

	print "number: ", number
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


def plot_image_changes(N=20, radius=5):
	orig_mat = create_random_colour_matrix(400,400)
	for i in xrange(N):
		orig_mat = matrix_average_step(orig_mat, radius)
		plt.imshow(orig_mat)
		plt.show()
	return orig_mat


if __name__ == '__main__':
	plot_image_changes()
