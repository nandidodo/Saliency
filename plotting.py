import matplotlib.pyplot as plt
from utils import *
from experiments import *
from tests import *

def show_img_red_green(img_fname, indices):
	#get imgs
	imgs = load_array(img_fname)
	#get our images we want to show
	imgs_to_show = []
	N = len(indices)
	for i in xrange(N):
		imgs_to_show.append(imgs[indices[i]])
	reds = []
	greens = []
	for j in xrange(N):
		img = imgs_to_show[i]
		reds.append(img[:,:,0])
		greens.append(img[:,:,1])
	
	for k in xrange(N):
		#full img
		plt.subplot(k+1, 3,1)
		plt.imshow(imgs_to_show[k])
		plt.title('original image')
		plt.xticks([])
		plt.yticks([])
		#red img
		plt.subplot(k+1, 3,2)
		plt.imshow(reds[k],cmap='gray')
		plt.title('red compoment')
		plt.xticks([])
		plt.yticks([])
		#green image
		plt.subplot(k+1, 3,3)
		plt.imshow(greens[k], cmap='gray')
		plt.title('green component')
		plt.xticks([])
		plt.yticks([])

	print "finished function!?"
	plt.show()	
if __name__ == '__main__':
	img_fname='testimages_combined'
	indices = (40,45,56)
	show_img_red_green(img_fname, indices)
