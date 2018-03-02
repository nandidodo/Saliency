import numpy as np
import cPickle as pickle
#I need to figure out where to import the test images from. Oh well
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imresize


def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


#get the panorama img
pan_fname ="./panorama_test_images/016.jpg"
#load it
pan_img = imread(pan_fname)
#save it for later ease of use
save_fname = "pan_img"
save_array(pan_img, save_fname)
plt.imshow(pan_img)
plt.show()
print pan_img.shape


