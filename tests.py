# this is where I just put really random things which are easy to find here. so we can inspect data and so forth. this isgoig to be a very freeform file. for scripting, as that's what python is meant to do.

import numpy as np
import scipy
from file_reader import *
from utils import *
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10


img = read_image(20)
#print img.shape
#show_colour_splits(img)
"""
# okay, let's try out the fft stuff
img_freq = np.fft.fft2(img)
#calculate amplitude spectum
img_amp = np.fft.fftshift(np.abs(img_freq))
#for display take the logarithm so the scale isn't so small
img_amp_display = np.log(img_amp + 0.0001)
#rescale to -1:+1 for display
img_amp_display = (((img_amp_display - np.min(img_amp_display))*2)/np.ptp(img_amp_display)) -1

print type(img_amp_display)
print img_amp_display.shape

print img
print "  "
print "  "
print img_amp_display

img_amp_display = img_amp_display *255
img_amp_display =  img_amp_display.astype('uint8')
plt.imshow(img_amp_display)
plt.show()"""

"""

# let's test our high pass filtering
#get_amplitude_spectrum(img, show=True)
#get_magnitude_spectrum(img, show=True)
# reshape
print img.shape
# okay, let's get this as grayscale
img = img[:,:,0]
img = reshape_into_image(img)
print img.shape

hpf = high_pass_filter(img, show=True)
#print hpf

# okay, other band tests here:
plt.imshow(img)
plt.show()
print "High Pass!"
hpf = highpass_filter(img, show=True)
print "  "
print "Low Pass"
lpf = lowpass_filter(img, show=True)
print "  "
print "Bandpass"
bpf = bandpass_filter(img, show=True)

print "both"
compare_two_images(lpf, hpf)
"""

# okay, we're going to test the filesystem stuff here
rootdir = './testSet/Stimuli'
#print_dirs_files(rootdir)
save_images_per_directory(rootdir, save=False)




#plt.imshow(img_amp_display)
#plt.show()
"""
error_maps = load('error_map_test')
error_map = error_maps[1]
print error_map.shape
err = np.reshape(error_map, [28,28])
print err.shape
plt.imshow(err)
plt.show()"""

"""
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
print xtrain.shape

red = xtrain[:,:,:,0]
print red.shape
redimg = red[5]
print redimg.shape


plt.imshow(xtrain[5])
plt.show()

plt.imshow(redimg)
plt.show()
"""






"""
a = [[1,5,3],[4,5,7],[7,5,2]]
a = np.array(a)
print a
print a.shape
print np.argmax(a)
print np.argmax(a, axis=0)
print np.argmax(a, axis=1)"""

# okay, fuck it. I'm going to search the entire array.this is going to take for fucking ever, and be totally terrible, but I'm going to do it, because I don't understnad what the fuck numpy argmax is actually doing?



