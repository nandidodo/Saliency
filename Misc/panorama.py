import numpy as np
import scipy
import Math

#so how this works is that there is a big image, and a viewport
# this just deals with the various image transformations needed to view the other image
#it's all done in coordinates, I think so this is just a couple of the image 
#movement features which should make sense, a couple of the functions that returns
# a new image when required!

def move_viewport(panorama_img, new_centre, viewport_width, viewport_height, edge_func=pad_edge):
	assert len(panorama_img.shape)==2, 'Panorama image must be two dimensional'
	pan_width, pan_height = panorama_img.shape
	assert viewport_width >0 and viewport_width <=pan_width, 'viewport width must be greater than zero and less or equal to the covering panoramic image width'
	assert viewport_height > 0 and viewport_height <=pan_height,'viewport height must be greater than zero and less or equal to the covering panoramic image height'
	assert len(new_centre)==2, 'New centre dimensions must also be two_dimensional'
	new_width, new_height = new_centre
	assert new_width >0 and new_width <=pan_width, 'New width must be within the covering panoramic image'
	assert new_height > 0 and new_height <=pan_height,'New height must be within the covering panoramic image'
	
	#asserts done, let's move onto the main method
	vw = viewport_width//2
	vh = viewport_height//2

	left_overextension = 0
	right_overextension = 0
	top_overextension = 0
	bottom_overextension =0

	if new_width -vw <0:
		left_overextension = np.abs(new_width - vw)

	if new_width + vw >pan_width:
		right_overextension = (new_width + vw)-pan_width

	if new_height + vh > pan_height:
		top_overextension = (new_height + vh) - pan_height
	
	if new_height -vh < 0:
		bottom_overextension = np.abs(new_height - vh)

	#assert there haven't been any strange errors in which there are negative overextensions
	assert left_overextension >=0, 'Left overextension cannot be negative. Check this'
	assert right_overextension >=0, 'Right overextension cannot be negative. Check this'
	assert top_overextension >=0, 'Top overextension cannot be negative. Check this'
	assert bottom_overextension >=0, 'Bottom overextension cannot be negative. Check this'

	if left_overextension ==0 and right_overextension==0 and top_overextension==0 and bottom_overextension==0:
		#no edges reached so it is easy to return
		new_img = panorama_img[new_width-vw: new_width+vw][new_height-vh: new_height+vh]
		#hopefully this will work
		return new_img
	else:
		#this is where it gets nasty... dagnabbit
		new_img = edge_func(panorama_img, new_centre, viewport_width,viewport_height,left_overextension, right_overextension, top_overextension, bottom_overextension)
		return new_img
		


def pad_edge(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	w,h = panorama.shape
	nw, nh = centre
	vw = viewport_width//2
	vh = viewport_height//2
	#setup the base image as all zeros, as that's the padding!
	new_img = np.zeros((viewport_width, viewport_height))
	#I could do this as a giant loop, but figuring out the array slice notation
	#is probably better!
	new_img[left:viewport_width-right][bottom:viewport_height-top]=panorama[nw-vw+left:nw+vw-right][nh-vh+bottom:nh+vh-top]
	return new_img
	#that wasn't actually that nasty at all, which is nice!
	#the wrap is going to be significantly more difficult, dagnabbit!
	#but I'm going to just ignore that for now... yay!


def pad_vertical_wrap_horizontal(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	w,h = panorama.shape
	assert len(centre)==2,'Centre point must be two dimensional'
	nw,nh = centre
	vw = viewport_width//2
	vh = viewport_width//2
	#initialize with zeros
	new_img = np.zeros((viewport_width, viewport_height))
	#If top or bottom is expanding need to expand the panorama to deal with this first
	if bottom>0 or top>0:
		#save old panorama
		old_panorama = panorama
		#create larger, zeroed panorama
		panorama = np.zeros((w, h + bottom + top))
		#situate the actual panorama within this!
		panorama[0:w][bottom:bottom+h] = old_panorama[0:w][0:h]
	#now that top/bottom are wrapped fill in the panorama
	new_img[left:viewport_width-right][bottom:viewport_height-top] = panorama[nw-vw+left:nw+vw-right][nh-vh+bottom:nh+vh-top]
	#now begin the wrap
	if left>0:
		new_img[0:left][:] = panorama[0:left][:]
	if right>0:
		new_img[vh-right:vh][:] = panorama[h-right:h]

	#hopefully this should be enough, so return!
	return new_img
