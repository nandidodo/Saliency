import numpy as np
import scipy 
import matplotlib.pyplot as plt
from spatial_frequencies import *
from log_polar import *


def pad_edge(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	assert len(panorama.shape)==2, 'Panorama image must be two dimensional'
	h,w = panorama.shape
	assert len(centre)==2, 'Centre point must be two dimensional'
	nh, nw = centre
	vw = viewport_width//2
	vh = viewport_height//2

	#setup the base image as all zeros, as that's the padding!
	new_img = np.zeros((viewport_height, viewport_width))
	#fill in the panorama image
	new_img[bottom:viewport_height-top, left:viewport_width-right] = panorama[nh-vh+bottom:nh+vh-top, nw-vw+left:nw+vw-right]
	return new_img


def pad_vertical_wrap_horizontal(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	assert len(panorama.shape)==2, 'Panorama image must be two dimensional'
	h,w= panorama.shape
	assert len(centre)==2,'Centre point must be two dimensional'
	nh, nw = centre
	vw = viewport_width//2
	vh = viewport_width//2
	#initialize with zeros
	new_img = np.zeros((viewport_height, viewport_width))
	#If top or bottom is expanding need to expand the panorama to deal with this first
	if bottom>0 or top>0:
		#save old panorama
		#old_panorama = np.copy(panorama)
		#create larger, zeroed panorama
		panorama = np.zeros((h+bottom+top, w))
		#situate the actual panorama within this!
		panorama[bottom:bottom+h, 0:w] = old_panorama[0:h, 0:w]
	#now that top/bottom are wrapped fill in the panorama
	new_img[bottom:viewpot_height-top, left:viewport_width-right] = panorama[nh-vh+bottom: nh+vh-top, nw-vw+left:nw+vw-right]
	#now begin the wrap
	if left>0:
		new_img[:, 0:left] = panorama[:, w-left: w]
	if right>0:
		new_img[:, vh-right:vh] = panorama[:, 0:right]
	return new_img

def wrap_horizontal_and_vertical(panorama, centre, viewport_width, viewport_height, left, right, top, bottom):
	assert len(panorama.shape)==2, 'Panorama image must be two dimensional'
	h,w = panorama.shape
	assert len(centre)==2,' Centre point must be two dimensional'
	nh,nw = center
	vw = viewport_width//2
	vh = viewport_width//2
	#initialise the new viewport with zeros
	new_img = np.zeros((viewport_height, viewport_width))
	#first fill in the panorama as best I can
	new_img[bottom:viewport_height-top, left:viewport_width-right] = panorama[nh-vh+bottom:nh+vh-top, nw-vw+left:nw+vw-right]
	#begin the wrap
	if left>0:
		new_img[:, 0:left] = panorama[:,w-left:w]
	if right>0:
		new_img[:, vh-right:vh] = panorama[:, 0:right]
	if top>0:
		new_img[h-top:h, :] = panorama[0:top,:]
	if bottom>0:
		new_img[0:bottom,:] = panorama[h-bottom:h,:]
	return new_img



def highlight_viewport(panorama_img, centre, viewport_width, viewport_height, border_width=10):
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
	#copy the array since it is modifying in place and I don't want to mutate the original panorama image!
	highlighted_img = np.copy(panorama_img)

	#initialise values for overrun checks
	vhb = vh
	vht = vh
	vwl = vw
	vwr = vw

	#check for overruns into padding. If so the borders need to be cropped to fit within the truncated image
	if ch + vh > nh:
		vht = nh-ch-border_width
	if ch -vh < 0:
		vhb = ch - border_width
	if cw - vw <0:
		vwl = cw - border_width
	if cw + vw >nw:
		vwr = nw-cw - border_width

	#now modify the image
	#left slice
	highlighted_img[ch-vhb:ch+vht, cw-vwl-border_width:cw-vwl] = np.full((vht + vhb,border_width), max_val)
	#right slice
	highlighted_img[ch-vhb:ch+vht, cw+vwr:cw+vwr+border_width] = np.full((vht+vhb,border_width),max_val)
	#top slice
	highlighted_img[ch+vht:ch+vht+border_width, cw-vwl:cw+vwr] = np.full((border_width, vwl+vwr),max_val)
	#bottom slice
	highlighted_img[ch-vhb-border_width:ch-vhb, cw-vwl:cw+vwr] = np.full((border_width,vwl+vwr), max_val)
	
	return highlighted_img

def center_surround_low_spatial_frequency(pan_img, viewport_centre, viewport_width, viewport_height, already_lowshifted=False):
	assert len(pan_img.shape)==2, 'Panorama must be two dimensional'
	assert len(viewport_centre)==2, 'Viewport center point must be two dimensional'
	h,w = pan_img.shape
	ch, cw = viewport_centre
	assert viewport_width > 0 and viewport_width <= w, 'Viewport width must be greater than 0 and less than the covering panoramic image width'
	assert viewport_height > 0 and viewport_height <= w, 'Viewport height must be greater than 0 and less than the covering panoramic image height'
	
	vw = viewport_width//2
	vh = viewport_width//2

	#copy since I'm doing in place mutations of the image
	new_img = np.copy(pan_img)

	if not already_lowshifted:
		#do the low spatial frequency transformation
		#do it
		new_img = lowpass_filter(new_img)

	#now add in viewport
	#the only thing I can think of is that one is normalised and the other is not
	#yeah, that's it!
	#normalise pan_img
	pan_img = pan_img.astype('float32')/255.
	new_img[ch-vh:ch+vh, cw-vw:cw+vw] = pan_img[ch-vh:ch+vh, cw-vw:cw+vw]
	return new_img


def center_surround_low_spatial_frequency_circular(pan_img, viewport_center, viewport_radius, already_lowshifted=False):
	assert len(pan_img.shape)==2, 'Panorama image must be two dimensional'
	h,w = pan_img.shape
	assert len(viewport_center.shape)==2, 'Viewport center must be two dimensional also'
	center_height, center_width = viewport_center
	assert center_height>=0 and center_height<=h, 'Viewport width must be within panorama image'
	assert center_width>=0 and center_width<=w, 'Viewport width must be within panorama image'
	assert viewport_radius<=h and viewport_radius<=w, 'Viewport radius must be within image dimensions'
	#get copy so as not to mutate new img in memory!
	new_img = np.copy(pan_img)

	if not already_lowshifted:
		#do the low spatial frequency transformation
		#do it
		new_img = lowpass_filter(new_img)

	#go through the image sequentially and check euclid distance
	for i in xrange(h):
		for j in xrange(w):
			dist = euclidean_distance((i,j), viewport_center)
			if dist<=viewport_radius:
				new_img[i][j] = pan_img[i][j]

	return new_img


def center_surround_low_spatial_frequency_multiple_stripes(pan_img, viewport_center, viewport_radius, blend_region_radius, stripe_radius, peripheral_blur,stripe_decline=None):
	assert len(pan_img.shape)==2, 'Panorama image must be two dimensional'
	h,w = pan_img
	assert len(viewport_center.shape)==2,'Viewport center point must be two dimensional'
	ch,cw = viewport_center
	assert viewport_radius>0 and viewport_radius <=h and viewport_radius<=w, 'Viewport must fit inside of panorama image'	
	assert blend_region_radius>0 and blend_region_radius<=h and blend_region_radius<=w,' Blend region must be reasonably sized'
	assert stripe_radius>0 and stripe_radius<=blend_region_radius,'Stripe radius cannot be larger than blend region radius'
	if stripe_decline is not None:
		assert stripe_decline >0, 'Stripe decline must be a positive number'

	max_resolution = 40
	
	#okay, begin algorithm
	#first copy image
	new_img = np.copy(pan_img)
	#lowhisft for periphery
	new_img = lowpass_filter(new_img)
	#add in foveated region
	#do as a horrendously inefficient for loop!
	for i in xrange(h):
		for j in xrange(w):
			dist = euclidean_distance((i,j), viewport_center)
			if dist<=viewport_radius:
				new_img[i][j] = pan_img[i][j]

	#now calculate the blend region radius and stripe radius
	num_stripes = blend_region_radius//stripe_radius
	#if stripe decline is none calculate it
	if stripe_decline is None:
		stripe_decline = (max_resolution - peripheral_blur)/num_stripes
		if stripe_decline ==0:
			#just in case it rounds too low, there is some stripe decline
			stripe_decline=1

	#okay, now begin the blend region
	for k in xrange(num_stripes):
		#I really shuold not have to copy and lowshift the entire image to do this
		#but I'm going to
		lowshifted = np.copy(pan_img)
		lowshifted = lowpass_filter(lowshifted, max_resolution - (k*stripe_decline))
		#then do the loop
		for i in xrange(h):
			for j in xrange(w):
				dist = euclidean_distance((i,j), viewport_center)
				if dist >=viewport_radius + ((k-1)*stripe_radius and dist <=viewport_radius + k*stripe_radius:
					new_img[i][j]=lowshifted[i][j]
	return new_img
				
			
	
	
	



def move_viewport_log_polar_transform(pan_img, viewport_center):
	assert len(pan_img.shape)==2, 'Panorama image must be two dimensional'
	h,w = pan_img.shape
	assert len(viewport_center)==2, 'Viewport center must be a two dimensional point'
	ch, cw = viewport_center
	assert ch>=0 and ch<=w, 'Viewport center height must be within the panorama image boundaries'
	assert cw>=0 and cw<=w, 'Viewport center width must be within the panorama image boudnaries'
	return logpolar_fancy(pan_img, ch,cw)




def get_pan_indices_of_viewport_centre(indices, viewport_centre, viewport_width, viewport_height, pan_width, pan_height):
	assert len(indices)==2,'Centre index must be of length 2'
	assert len(viewport_centre)==2, 'Viewport centre (in pan img coordinates) must be two dimensional'
	#what if it is outside of the panorama img?
	assert viewport_width > 0 and viewport_width < pan_width, 'Viewport width must be greater than 0 and less than panorama width'
	assert viewport_height > 0 and viewport_height < pan_height, 'Viewport height must be greater than 0 and less than panorama width'
	assert pan_width > 0,'Panorama width must be greater than 0'
	assert pan_height > 0, 'Panorama height must be greater than 0'
	#don't let it get so
	vh = viewport_height//2
	vw = viewport_width//2

	ch, cw = indices
	vch, vcw = viewport_centre

	pan_w = vcw - vw + cw
	pan_h = vch - vh +ch

	#stop at edge if overshoots
	if pan_w <0:
		pan_w = 0
	if pan_w > pan_width:
		pan_w = pan_width

	if pan_h<0:
		pan_h = 0
	if pan_h>pan_height:
		pan_h = pan_height
	#because this is how the viewport is defined, right?
	#if this works it will be a miracle!
	return (pan_h, pan_w)



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
	plt.imshow(panorama, cmap=cmap)
	plt.title('Panorama Image')
	plt.xticks([])
	plt.yticks([])

	ax2 = fig.add_subplot(122)
	plt.imshow(viewport, cmap=cmap)
	plt.title('Viewport Image')
	plt.xticks([])
	plt.yticks([])
	
	plt.show()
	return fig
	
		



