# okay, let's try a log polar sampling. This should be really fun. I'm really not sure how to recreate the image in a way we can display it though. that's realy bad... ugh!
# this does seem to be extremely important as it's how the brain actually perceives these thigns
# and writing it up seems like a very valuable thing to be doing,so we can understnad just how it works
# and also understand just how julia works also, as a language, as writing the algorithm in that first, although in python also seems to be very important


## let's try this as a very very simple loop. it'll be death slow in python, but that won'tbe too horrendous

# I wonder actually let's define a really simple and hacky transform which just blitzes the resolutoin of the thing out so it makes sense, with the resolution falling off at each time. I honestly don't know. also, we need to know what to do if we reach edges and so forth, but I don't know... argh. dagnabbit edge effects. let's try out a cool image transform and see what happens this is just artificially decreasing the resolution to test and see if anything interesting is happening

# ythis is just a file where I try to figure out some simple python foveal image transforms to see what's up here, because this will undoubtedly be very useful and cool. ew can do this... let's go!

import numpy as np
import matplotlib.pyplot as plt
from utils import *

# let's just import an image
imgs = load_array('testimages_combined')
print imgs.shape
img = imgs[40]

# okay, I forgot about multiple channels. let's just grayscale this
#img = img[:,:,0]
#img = np.reshape(img, (img.shape[0], img.shape[1]))
plt.imshow(img)
plt.show()

# okay great. I forgot they were only doing stuff on the 100/100 iamges. oh well. we'll have to fix that later, but for now it doesn't matter. let's first just figure out a simple manual for loop thing that decreases the resolution at each tick of the image from the centre

#not sure to the extent to which the resolution should decrease. I really have no clue. guess I should find out, but let's test it for a bit

# this is going to be horrendously inefficient, as it is python, and not vectorised code

#okay, our very simple and currently primitive distance function

# okay, well, thisdoesn' really work, but nevertheless it's cool. I should also go and have a shower now, I think
def resolution_by_distance(dist):
	if dist<=20:
		return 0
	#if dist>20:
	#	return int(dist/30)
	if dist > 20 and dist <40:
		return 1
	if dist > 40:
		return 2


def manual_resolution_decrease(img, centre):
	shape = img.shape
	lenx = shape[0]
	leny = shape[1]
	centrex = centre[0]
	centrey = centre[1]

	#init our new img
	newimg = np.zeros(shape)

	#do asserts
	assert centrex > 0 and centrex <lenx, 'x coordinate of centre outside image'
	assert centrey > 0 and centrey <leny, 'x coordinate of centre outside image'

	for i in xrange(lenx):
		for j in xrange(leny):
			# we calculate distance from centre
			dist = np.sqrt((i-centrex)**2 + (j-centrey)**2)
			# we have soem distance function which maps to numbers of boxes to average over?
			res = resolution_by_distance(dist)
			print res
			#we initialise total and num with current position
			total = img[i][j]
			num = 1
			if res > 0:
				for k in xrange(res):
					for l in xrange(res):
						propx = i-k
						propy=j-l
						#if it's within boundaries
						if propx >-1 and propx <lenx and propy>-1 and propy<leny:
							total += img[propx][propy]
							num +=1
						propx = i+k
						propy=j+l
						#if it's within boundaries
						if propx >-1 and propx <lenx and propy>-1 and propy<leny:
							total += img[propx][propy]
							num +=1
			avg = float(total)/float(num)
			newimg[i][j]=avg
	return newimg


#centre = (50, 50)
#fovimg = manual_resolution_decrease(img, centre)
#plt.imshow(fovimg,cmap='gray')
#plt.show()


# okaty, I copied this code from stackoverflow. let's see if it works at all

def polar2cart(r, theta, center):

    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center, final_radius, initial_radius = None, phase_width = 99):

    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    print Xcart
    print Ycart
    print Xcart.shape
    print Ycart.shape

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img

#centre = (50,50)
#final_radius = 80
#fovimg = img2polar(img, centre, final_radius)
#plt.imshow(fovimg, cmap='gray')
#plt.show()



def _lpcoords(ishape, w, angles=None):
    """Calculate the reverse coordinates for the log-polar transform.
    Return array is of shape (len(angles), w)
    """

    ishape = np.array(ishape)
    bands = ishape[2]

    oshape = ishape.copy()
    centre = (ishape[:2]-1)/2.

    d = np.hypot(*(ishape[:2]-centre)) # maximum radius
    log_base = np.log(d)/w

    if angles is None:
        angles =  -np.linspace(0, 2*np.pi, 4*w + 1)[:-1]
    theta = np.empty((len(angles),w),dtype=np.float64)
    # Use broadcasting to replicate angles
    theta.transpose()[:] = angles

    L = np.empty_like(theta)
    # Use broadcasting to replicate distances
    L[:] = np.arange(w).astype(np.float64)

    r = np.exp(L*log_base)

    return r*np.sin(theta) + centre[0], r*np.cos(theta) + centre[1], \
angles, log_base

## okay, well their code doesn't work either. let's try this one
def logpolar(image, angles=None, Rs=None, mode='M', cval=0, output=None,
             _coords_r=None, _coords_c=None, extra_info=False):
    """Perform the log polar transform on an image.
    Input:
    ------
    image : MxNxC array
        An MxN image with C colour bands.
    Rs : int
        Number of samples in the radial direction.
    angles : 1D array of floats
        Angles at which to evaluate. Defaults to 0..2*Pi in 4*Rs steps
        ([1] below suggests 8*Rs, but that causes too much oversampling).
    mode : string
        How values outside borders are handled. 'C' for constant, 'M'
        for mirror and 'W' for wrap.
    cval : int or float
        Used in conjunction with mode 'C', the value outside the border.
    extra_info : bool
        Whether to return the angles and log base in addition
        to the transform.  False by default.
    Returns
    -------
    lpt : ndarray of uint8
        Log polar transform of the input image.
    angles : ndarray of float
        Angles used.  Only returned if `extra_info` is set
        to True.
    log_base : int
        Log base used.  Only returned if `extra_info` is set
        to True.
    Optimisation parameters:
    ------------------------
    _coords_r, _coords_c : 2D array
        Pre-calculated coords, as given by _lpcoords.
    References
    ----------
    .. [1] Matungka, Zheng and Ewing, "Image Registration Using Adaptive
           Polar Transform". IEEE Transactions on Image Processing, Vol. 18,
           No. 10, October 2009.
    """
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError("Input image must be 2 or 3 dimensional.")

    image = np.atleast_3d(image)

    if Rs is None:
        Rs = max(image.shape[:2])

    if _coords_r is None or _coords_c is None:
        _coords_r, _coords_c, angles, log_base = \
                   _lpcoords(image.shape, Rs, angles)

    bands = image.shape[2]
    if output is None:
        output = np.empty(_coords_r.shape + (bands,),dtype=np.uint8)
    else:
        output = np.atleast_3d(np.ascontiguousarray(output))
    for band in range(bands):
        output[...,band] = interp_bilinear(image[...,band],
                                           _coords_r,_coords_c,mode=mode,
                                           cval=cval,output=output[...,band])

    output = output.squeeze()

    if extra_info:
        return output, angles, log_base
    else:
		return output

#fovimg = logpolar(img)
#plt.imshow(fovimg, cmap='gray')
#plt.show()

# okay, we're going to try another different polar conversion t see if this works
import numpy as np
from scipy.ndimage.interpolation import geometric_transform

def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order. 
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal 
    # from a corner to the mid-point of img.

    print "in to polar"
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        return i,j

    print "beginning sklearn geometric transform"
    polar = geometric_transform(img, transform, order=order)
    print "finished geometric transform"

    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)


from skimage.data import chelsea

print "stating end function"

import matplotlib.pyplot as plt

img = chelsea()[...,0] / 255.
pol, (rads,angs) = topolar(img)

print "finished to polar"

fig,ax = plt.subplots(2,1,figsize=(6,8))

ax[0].imshow(img, cmap=plt.cm.gray, interpolation='bicubic')

ax[1].imshow(pol, cmap=plt.cm.gray, interpolation='bicubic')

ax[1].set_ylabel("Radius in pixels")
ax[1].set_yticks(range(0, img.shape[0]+1, 50))
ax[1].set_yticklabels(rads[::50].round().astype(int))

ax[1].set_xlabel("Angle in degrees")
ax[1].set_xticks(range(0, img.shape[1]+1, 50))
ax[1].set_xticklabels((angs[::50]*180/3.14159).round().astype(int))

plt.show()

