# just a quick and dirty script to resize these images to 100x100 for the gestalt network to work

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imresize
import os


def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


def resize(fname, resize_pix):
	img_mode = 'RGB'
	temp_fname = "temp.png"
	imgs = load_array(fname)
	imgs = imgs.astype('float32')/255.
	newimgs = []
	for i in xrange(len(imgs)):
		#img = plt.imshow(imgs[i])
		img = imgs[i]
		plt.imsave(temp_fname, img)
		img = imresize(imread(temp_fname, mode='RGB'), resize_pix)
		newimgs.append(img)
	newimgs = np.array(newimgs)
	#delete the temp final one
	os.remove(temp_fname)
	return newimgs



if __name__ == '__main__':
	#newimgs = resize("BenchmarkIMAGES_images", (100,100))
	#save_array(newimgs, "BenchmarkIMAGES_images_resized_100x100")
	testimgs = load_array("BenchmarkIMAGES_images_resized_100x100")
	print testimgs.shape
	for i in xrange(10):
		plt.imshow(testimgs[i])
		plt.show()
		
	
