import numpy as np
import scipy 
import matplotlib.pyplot as plt


def pad_edge(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	w,h = panorama.shape
	nw, nh = centre
	vw = viewport_width//2
	vh = viewport_height//2
	#setup the base image as all zeros, as that's the padding!
	new_img = np.zeros((viewport_height, viewport_width))
	#I could do this as a giant loop, but figuring out the array slice notation
	#is probably better!
	#new_img[left:viewport_width-right, bottom:viewport_height-top]=panorama[nw-vw+left:nw+vw-right, nh-vh+bottom:nh+vh-top]
	new_img[bottom:viewport_height-top, left:viewport_width-right] = panorama[nh-vh+bottom:nh+vh-top, nw-vw+left:nw+vw-right]
	return new_img
	#that wasn't actually that nasty at all, which is nice!
	#the wrap is going to be significantly more difficult, dagnabbit!
	#but I'm going to just ignore that for now... yay!


def pad_vertical_wrap_horizontal(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	w,h = panorama.shape
	assert len(centre)==2,'Centre point must be two dimensional'
	nh, nw = centre
	vw = viewport_width//2
	vh = viewport_width//2
	#initialize with zeros
	new_img = np.zeros((viewport_height, viewport_width))
	#If top or bottom is expanding need to expand the panorama to deal with this first
	if bottom>0 or top>0:
		#save old panorama
		old_panorama = panorama
		#create larger, zeroed panorama
		#panorama = np.zeros((w, h + bottom + top))
		panorama = np.zeros((h+bottom+top, w))
		#situate the actual panorama within this!
		#panorama[0:w, bottom:bottom+h] = old_panorama[0:w, 0:h]
		panorama[bottom:bottom+h, 0:w] = old_panorama[0:h, 0:w]
	#now that top/bottom are wrapped fill in the panorama
	#new_img[left:viewport_width-right, bottom:viewport_height-top] = panorama[nw-vw+left:nw+vw-right, nh-vh+bottom:nh+vh-top]
	new_img[bottom:viewpot_height-top, left:viewport_width-right] = panorama[nh-vh+bottom: nh+vh-top, nw-vw+left:nw+vw-right]
	#now begin the wrap
	if left>0:
		#new_img[0:left, :] = panorama[0:left, :]
		new_img[:, 0:left] = panorama[:, w-left: w]
	if right>0:
		#new_img[vh-right:vh, :] = panorama[h-right:h, :]
		new_img[:, vh-right:vh] = panorama[:, 0:right] #hopefully this shold work

	#hopefully this should be enough, so return!
	return new_img

def wrap_horizontal_and_vertical(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	h,w = panorama.shape
	assert len(centre)==2,' Centre point must be two dimensional'
	nw, nh = center
	vw = viewport_width//2
	vh = viewport_width//2
	#initialise the new viewport with zeros
	new_img = np.zeros((viewport_height, viewport_width))
	#first fill in the panorama

	#new_img[left:viewport_width-right,bottom:viewport_height-top] = panorama[nw-vw+left:nw+vw-right,nh-vh+bottom:nh+vh-top]
	#new_img[bottom:viewport_height-top, left:viewport_width-right] = panorama[nh-vh+bottom:nh+vh-top, nw-vw+left:nw+vw-right]
	#begin the wrap
	if left>0:
		#new_img[0:left,:] = panorama[w-left:w, :]
		new_img[:, 0:left] = panorama[:,w-left:w]
	if right>0:
		#new_img[vh-right:vh, :] = panorama[0:right, :]
		new_img[:, vh-right:vh] = panorama[:, 0:right]
	if top>0:
		#new_img[:, h-top:h] = panorama[:, 0:top]
		new_img[h-top:h, :] = panorama[0:top,:]
	if bottom>0:
		#new_img[:, 0:bottom] = panorama[:, h-bottom:h]
		new_img[0:bottom,:] = panorama[h-bottom:h,:]
	return new_img



def highlight_viewport(panorama_img, centre, viewport_width, viewport_height, border_width=10):
	#this just returns the panorama img with the viewport highlighted
	#I should probably do it as part of the other function, but I don't need to have
	#it should be easy
	#let's do asserts
	assert len(panorama_img.shape)==2, 'Panorama image must be two dimensional'
	nh, nw = panorama_img.shape
	assert len(centre)==2, 'Centre point must be two dimensional'
	ch, cw = centre
	assert viewport_width <=nw and viewport_width >0, 'Viewport width cannot be 0 or larger than the panorama image'
	assert viewport_height<=nh and viewport_height>0, 'Viewport height cannot be 0 or larger than the panoram image'
	assert ch >=0 and ch<=nh, 'Centre height must be within panorama image'
	assert cw>=0 and cw<=nw, 'Centre width must be within panorama image'

	assert type(border_width)==int and border_width>0 and border_width<viewport_width and border_width < viewport_height, 'Border width must be a positive integer and not larger than the viewport (which would be silly!)'

	vh = viewport_height//2
	vw = viewport_height//2

	#get max value
	max_val = np.amax(panorama_img)
	#left slice
	panorama_img[ch-vh:ch+vh, cw-vw-border_width:cw-vw] = np.full((viewport_height,border_width), max_val)
	#right slice
	panorama_img[ch-vh:ch+vh, cw+vw:cw+vw+border_width] = np.full((viewport_height,border_width),max_val)
	#top slice
	panorama_img[ch+vh:ch+vh+border_width, cw-vw:cw+vw] = np.full((border_width, viewport_width),max_val)
	#bottom slice
	panorama_img[ch-vh-border_width:ch-vh, cw-vw:cw+vw] = np.full((border_width,viewport_width), max_val)
	
	#sorted!
	return panorama_img


#so how this works is that there is a big image, and a viewport
# this just deals with the various image transformations needed to view the other image
#it's all done in coordinates, I think so this is just a couple of the image 
#movement features which should make sense, a couple of the functions that returns
# a new image when required!

def move_viewport(panorama_img, new_centre, viewport_width, viewport_height, edge_func=pad_edge, move_outside_img=True,show_viewport=False):
	assert len(panorama_img.shape)==2, 'Panorama image must be two dimensional'
	pan_height, pan_width = panorama_img.shape
	assert viewport_width >0 and viewport_width <=pan_width, 'viewport width must be greater than zero and less or equal to the covering panoramic image width'
	assert viewport_height > 0 and viewport_height <=pan_height,'viewport height must be greater than zero and less or equal to the covering panoramic image height'
	assert len(new_centre)==2, 'New centre dimensions must also be two_dimensional'
	new_height, new_width = new_centre
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


	#If can't move outside image, check asserts that all overextensions are the same
	if move_outside_img==False:
		assert left_overextension==0, 'This would move the viewport outside the left of the image'
		assert right_overextenion==0, 'This would move the viewport outside the right of the image'
		assert top_overextension==0,'This would move the viewport above the top of the image'
		assert bottom_overextension==0, 'This would move the viewport below the bottom of the image'
	#assert there haven't been any strange errors in which there are negative overextensions
	assert left_overextension >=0, 'Left overextension cannot be negative. Check this'
	assert right_overextension >=0, 'Right overextension cannot be negative. Check this'
	assert top_overextension >=0, 'Top overextension cannot be negative. Check this'
	assert bottom_overextension >=0, 'Bottom overextension cannot be negative. Check this'

	if left_overextension ==0 and right_overextension==0 and top_overextension==0 and bottom_overextension==0:
		#no edges reached so it is easy to return
		print "no edges reached in viewport"
		print new_width-vw
		print new_width+vw
		print new_height-vh
		print new_height+vh
		new_img = np.zeros((viewport_height, viewport_width))
		#new_img = panorama_img[new_width-vw: new_width+vw, new_height-vh: new_height+vh]
		new_img = panorama_img[new_height-vh:new_height+vh, new_width-vw:new_width+vw]
		
		if show_viewport:
			return new_img, highlight_viewport(panorama_img,new_centre, viewport_width, viewport_height)

		return new_img
	else:
		#this is where it gets nasty... dagnabbit
		new_img = edge_func(panorama_img, new_centre, viewport_width,viewport_height,left_overextension, right_overextension, top_overextension, bottom_overextension)

		if show_viewport:
			return new_img, highlight_viewport(panorama_img,new_centre, viewport_width, viewport_height)

		return new_img




def show_panorama_and_viewport(panorama, viewport,figsize=None,cmap=None):
	
	assert len(panorama.shape)==2, 'Panorama image must be two dimensional'
	assert len(viewport.shape)==2, 'Viewport image must be two dimensional'

	if figsize is None:
		figsize=32

	if cmap is None:
		cmap='gray'

	fig = plt.figure(figsize)
	ax1 = fig.add_subplot(121)
	plt.imshow(panorama)
	plt.title('Panorama Image')
	plt.xticks([])
	plt.yticks([])

	ax2 = fig.add_subplot(122)
	plt.imshow(viewport)
	plt.title('Viewport Image')
	plt.xticks([])
	plt.yticks([])
	
	plt.show()
	return fig
	
		



