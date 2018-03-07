# okay, so the aim here is to have a bunch of functions which, in the end can try to mimic actual scanpaths, so I can see what's up there to be honest. They should deal with 2d image arrays and produce 

from __future__ import division
import numpy as np
import scipy
import matplotlib.pyplot as plt

def attempt_image_reshape_2d(img):
	sh = img.shape
	length = len(sh)
	if length == 2:
		return img
	if length ==1:
		print "Warning in the image reshape. Length should be at least 2. Attempting to reshape by square-rooting."
		dim = np.sqrt(sh[0])
		#assert it's an integer or something here
		try:
			return np.reshape(img, (dim, dim))
		except Exception as e:
				print "Cannot reshape the single dimension: " + str(e)
				return

	if length==3:
		if sh[2] == 1 or sh[0] ==1:
			if sh[2]==1:
				try: 
					return np.reshape(img, (sh[0], sh[1]))
				except Exception as e:
					print 'Cannot reshape img' + str(e)
					return
			if sh[0]==1:
				try:
					return np.reshape(img, (sh[1], sh[2]))	
				except Exception as e:
					print "Cannot reshape the single dimenion according to first dimension: " + str(e)
					return

		else:
				raise AssertionError('3D images cannot be losslessly converted to 2D. You must implement the transition manually if that is what you want')
				return

	if length==4:
		if sh[0] ==1 and sh[3] ==1:
			try:
				return np.reshape(img, (sh[1], sh[2]))
			except Exception as e:
				print "Cannot reshape 4d img with two one dimensions: " + str(e)
				return
		else:
			raise AssertionError('Cannot reshape a 4D image to 2D. You must manually fix this or check you are not passing in the wrong array')
			return
	if length > 4:
		raise AssertionError('Cannot reshape a ' + str(length) + 'D image to 2D. You must manually fix this or check you are not passing in the wrong array')
		return
	if length<1:
		raise AssertionError('Length of input array must be at least 1, and ideally 2')
		return
	raise AssertionError('Unknown failure in the reshape_image_to_2d method, please check its inputs')
	return


def get_max_indices(img, return_val=False):
	assert len(img.shape)==2, 'Image must be two dimensional. Please reshape if not'
	width, height = img.shape
	max_val = 0
	max_indices = (0,0)
	for i in range(width):
		for j in range(height):
			val = img[i][j]
			if val > max_val:
				max_val = val
				max_indices = (i,j)
	if return_val:
		return max_indices, max_val

	return max_indices
				

def simple_max_val_scanpath(img, N = -1, scanpaths=True):
	#this is all going to be horrendously inefficient, but hopefully it shouldn't matter thatm uch
	#as it's not like these are going to be running for any longer
	assert len(img.shape) ==2, 'Image must be two dimensional. Please reshape if not'
	width, height = img.shape
	total_dim = width*height
	combined_scanpath = np.zeros((width, height))
	scanpaths_flag = scanpaths
	if scanpaths_flag:
		scanpaths = []
	if N == -1:
		N = total_dim
	for run in range(N):
		i,j = get_max_indices(img)
		img[i][j] = 0
		combined_scanpath[i][j] = total_dim - run
		if scanpaths_flag:
			scanpath = np.zeros((width, height))
			scanpath[i][j] = 1
			scanpaths.append(scanpath)
	
	if scanpaths_flag:
		scanpaths = np.array(scanpaths)
		return combined_scanpath, scanpaths
	
	return scanpaths


def euclidean_distance(indices):
	#make work for any dimensions
	assert len(indices)==2, 'Only calculates euclidean distance in two dimensions'
	x,y = indices
	return np.sqrt(x**2 + y **2)

def apply_gaussian_to_point(val, indices, centre_indices, sigma=None):
	#dagnabbit, undoubtedly there's a numpy function which does this for me in a more efficient manner. But I'm not sure what it is! So I'll implement it like this for now

	assert len(indices)==2, 'Indices must be two dimensional'
	assert len(centre_indices)==2, 'Centre indices must be two dimensional'
	#define default sigma # assume default of 2
	if sigma is None:
		sigma = np.array([[5,0],[0,5]])
	if type(sigma)==int:
		sigma=np.array([[sigm, 0],[0,sigma]])
	

	#gauss_normalizer = (1/np.sqrt(2*np.pi*sigma))

	#convert indices to numpy arrays
	indices =np.array(indices)
	centre_indices = np.array(centre_indices)
	
	#gauss_exponent = -1 * np.dot((indices-centre_indices).T, np.dot(np.linalg.inv(sigma), (indices-centre_indices)))
	#gauss_diff = gauss_normalizer * np.exp(gauss_exponent)
	#new_val = val - (val*gauss_diff)
	new_val = val - (val* pdf_2d_gauss(indices, centre_indices, sigma))
	return new_val


def pdf_2d_gauss(x, mu, cov):
	#so x is a simple 1d vector o pints,
	#mu is the same
	#cov is also the same
	#reshapes
	if len(mu.shape)==1:
		mu = np.reshape(mu, (len(mu),1))
	if len(x.shape)==1:
		x = np.reshape(x, (len(x),1))
	

	assert mu.shape[0]>mu.shape[1] and x.shape[1]==1, 'mu must be a row vector'
	assert x.shape[0] > x.shape[1] and x.shape[1]==1, 'X mus be a row vector'
	assert x.shape[0] ==mu.shape[0], 'x and mu must be of same dimension'
	assert cov.shape[0]==cov.shape[1], 'Covariance matrix must be square'
	assert mu.shape[0]==cov.shape[0], 'Covariance matrix and mu must have same dimension'

	#start the math
	denom = 1 / ((2*np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2))
	numer = -0.5 * ((x-mu).T.dot(np.linalg.inv(cov)).dot(x-mu))
	return float(denom * np.exp(numer))
	

def gaussian_filter(img, sigma=2):
	assert len(img.shape)==2,'Image must be two dimensional for gaussian filtering'
	return scipy.ndimage.filters.gaussian_filter(img, sigma)


def apply_gaussian(img, centre, sigma= 2):
	assert len(img.shape)==2, 'Image must be two dimensional'
	width, height = img.shape
	for i in range(width):
		for j in range(height):
			img[i][j] = apply_gaussian_to_point(img[i][j], (i,j), centre)
	return img

def scanpaths_with_gaussian_inhibition(img, N= 10, sigma=5, scanpaths=True):
	assert len(img.shape)==2, 'Image must be two dimensional'
	width, height = img.shape
	if N == -1:
		N = width*height
	scanpaths_flag = scanpaths
	if scanpaths_flag:
		scanpaths  = []
	for run in range(N):
		max_i, max_j = get_max_indices(img)
		img = apply_gaussian(img, (max_i, max_j), sigma)
		if scanpaths_flag:
			scanpath = np.zeros((width, height))
			scanpath[max_i][max_j] = 1
			scanpath = gaussian_filter(scanpath, sigma)
			scanpaths.append(scanpath)
	
	if scanpaths_flag:
		scanpaths = np.array(scanpaths)
		return scanpaths
		
def plot_nine_scanpaths(img, sal,scanpath_func = scanpaths_with_gaussian_inhibition, sigma=2, cmap='gray'):
	assert len(img.shape)==2, 'Image must be two dimensional'
	assert len(sal.shape)==2,'Sal map must be two dimensional'
	assert img.shape == sal.shape, 'Image and salience map must have same dimensions'
	
	scanpaths = scanpath_func(sal, 7,sigma=sigma)
	
	fig = plt.figure()
	
	ax1 = fig.add_subplot(331)
	plt.imshow(img, cmap=cmap)
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(332)
	plt.imshow(sal, cmap=cmap)
	plt.title('Salience Map')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(333)
	plt.imshow(scanpaths[0], cmap=cmap)
	plt.title('First Fixation')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(334)
	plt.imshow(scanpaths[1], cmap=cmap)
	plt.title('Second fixation')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(335)
	plt.imshow(scanpaths[2], cmap=cmap)
	plt.title('Third Fixation')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(336)
	plt.imshow(scanpaths[3], cmap=cmap)
	plt.title('Fourth Fixation')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(337)
	plt.imshow(scanpaths[4], cmap=cmap)
	plt.title('Fifth Fixation')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(338)
	plt.imshow(scanpaths[5], cmap=cmap)
	plt.title('Sixth Fixation')
	plt.xticks([])
	plt.yticks([])

	ax1 = fig.add_subplot(339)
	plt.imshow(scanpaths[6], cmap=cmap)
	plt.title('Seventh Fixation')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	plt.show()
	
	#now get the images
	
#I relaly don't know what 
		

#not sure what else to do with this then... because the scan paths are actually sorted here, so it's difficult! I think this is a reasonable scan path algorithm
# the next steps are obviously to define an infrastructure for taking large panoramic images and padding them and/or showing them various things, OR wrapping them around, idk
#how I should do that in an easy way to be honest ,but it could be a relatively simple thing to do and to test out, so let's do that also, to see if it works!
# because that would allow us to have moving/roving images over a whole host of the image
# and perhaps get some more realistic style scan paths, or at the very least some kind of interesting results there
# so how is this going to work out overallthen. I'm really not sure. So there's going tobe a big image and a view port of less than the width of the big image, and all it's going to be able to do is move to the edge of the big image, and as it does it seems smaller amounts of the image, which could be very useful to see how the actual eye appears to do it, and I could even do a low spatial frequency filtering of that if I can make it fast enough, but I probably can't
