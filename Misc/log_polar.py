# this is meant to be a naive and slow list doing of log polar
# basically I think the best way to understand this is to implement it and try to understand it
# multiple times, and then eventually I will write a julia function that does it fast
#before applying it to the database

import numpy as np

def logpolar_naive(image, i_0, j_0, p_n=None, t_n=None):
	#what are these? I'm fairly confused already
	(i_n, j_n) = image.shape[:2] # so width and height presumably


	#i_c and j_c are thecenter of the image d_c is he distance from the transorms focus	
	i_c = max(i_0, i_n - i_0)
	j_c = max(j_0, j_n - j_n)
	d_c = (i_c **2 + j_c**2) **0.5
	
	if p_n == None:
		p_n = int(np.ceil(d_c))
	if t_n == None:
		t_n = j_n # t_n is the default image of transform diension

	#this is a scale factor which determines the size of each step along the transform
	p_s = np.log(d_c)/p_n
	t_s = 2.0*np.pi / t_n

	#the transform's pixels have same depth and type and height as the input's
	transformed = np.zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)

	#scans the transform across it's coordinate axis. at each step calculates the return valueto the cartesian coordinate system, and if the coordiantes fall within the boundaries of the input image, takes that cell's value into the transform

	for p in range(0, p_n):
		p_exp = np.exp(p * p_s)
		for t in range(0, t_n):
			t_rad = t*t_s
			i = int(i_0 + p_exp * np.sin(t_rad))
			j = int(j_0 + p_exp * np.cos(t_rad))

			if 0 <=i < i_n and 0 <= j < j_n:
				transformed[p,t] = image[i,j]
	return transformed

	#so this doesn't do any kind of actual interpolation

# I mean we can cache it, which is what he does. Still doesn't deal with the fact that
# it's horrendously slow, but that doesn't really matter. I'll rewrite into julia for speed, as could be fun!

_transforms = {}

def _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s):
	transform = _transforms.get((i_0, j_0, i_n,j_n, p_n, t_n))

	#if transform is not found
	if transform == None:
		i_k = []
		j_k = []
		p_k = []
		t_k = []

		#scans transform across it's coordinate axes, at east step calculates reverse transform back into cartesian coordinate system and if coordinates fall within bounds of the input image, records both coordiantes set

		for p in range(0, p_n):
			p_exp = np.exp(p*p_s)
			for t in range(0, t_n):
				t_rad = t*t_s
				i = int(i_0 + p_exp *np.sin(t_rad))
				j = int(j_0 + p_exp * np.cos(t_rad))
		
				if 0 <=i < i_n and 0<=j < j_n:
					i_k.append(i)
					j_k.append(j)
					p_k.append(p)
					t_k.append(t)

	#creates a fancy set of two indeces
		transform = ((np.array(p_k), np.array(t_k)), (np.array(i_k), np.array(j_k)))
		_transforms[i_0, j_0, i_n, j_n, p_n, t_n] = transform

	return transform



def logpolar_fancy(image, i_0, j_0, p_n=None, t_n=None):
    r'''Implementation of the log-polar transform based on numpy's fancy
        indexing.
    
        Arguments:
            
        image
            The input image.
        
        i_0, j0
            The center of the transform.
        
        p_n, t_n
            Optional. Dimensions of the output transform. If any are None,
            suitable defaults are used.
        
        Returns:
        
        The log-polar transform for the input image.
    '''
    # Shape of the input image.
    (i_n, j_n) = image.shape[:2]
    
    # The distance d_c from the transform's focus (i_0, j_0) to the image's
    # farthest corner (i_c, j_c). This is used below as the default value for
    # p_n, and also to calculate the iteration step across the transform's p
    # dimension.
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5
    
    if p_n == None:
        # The default value to p_n is defined as the distance d_c.
        p_n = int(np.ceil(d_c))
    
    if t_n == None:
        # The default value to t_n is defined as the width of the image.
        t_n = j_n
    
    # The scale factors determine the size of each "step" along the transform.
    p_s = np.log(d_c) / p_n
    t_s = 2.0 * np.pi / t_n
    
    
    # Recover the transform fancy index from the cache, creating it if not
    # found.
    (pt, ij) = _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s)

    # The transform's pixels have the same type and depth as the input's.
    transformed = np.zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)

    # Applies the transform to the image via numpy fancy-indexing.
    transformed[pt] = image[ij]
    return transformed


from time import clock
from sys import maxint
from scipy.misc import imread, imsave


def profile(f):
    image = imread('panorama_test_images/016.jpg')
    transformed = None
    t_max = 0
    t_min = maxint
    for i in range(0, 10):
        t_0 = clock()
        transformed = f(image, 127, 127)
        t_n = clock()
        
        t = t_n - t_0
        if t > t_max:
            t_max = t
        if t < t_min:
            t_min = t

    name = f.__name__
    imsave('%s.png' % name, transformed)
    print('Best and worst time for %s() across 10 runs: (%f, %f)' % (name, t_min, t_max))


if __name__ == '__main__':
    profile(logpolar_naive)
    profile(logpolar_fancy)
	#image = imread('panorama_test_images/016.jpg')
	#logpolar = logpolar_fancy(image, 

