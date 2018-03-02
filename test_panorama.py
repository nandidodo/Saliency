import numpy as np
import cPickle as pickle
#I need to figure out where to import the test images from. Oh well
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imresize
from panorama import *


def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


#get the panorama img
#pan_fname ="./panorama_test_images/016.jpg"
#load it
#pan_img = imread(pan_fname)
#save it for later ease of use
#save_fname = "pan_img"
#save_array(pan_img, save_fname)

save_fname = "pan_img"
pan_img = load_array(save_fname)
#turn to grayscale
sh = pan_img.shape
pan_img = np.reshape(pan_img[:,:,0], (sh[0], sh[1]))
plt.imshow(pan_img, cmap='gray')
plt.show()
print pan_img.shape
#now test this

vw = 100
vh = 100
viewport = move_viewport(pan_img, (500, 500), vw, vh)
print viewport.shape
plt.imshow(viewport)
plt.show()


