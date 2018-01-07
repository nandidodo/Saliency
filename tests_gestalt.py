# okay, here are tests for the gestalt stuff

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))



def plot_both_six_image_comparison(leftpreds, rightpreds, leftslice, rightslice, N=50):
	shape = leftpreds.shape
	assert shape == rightpreds.shape == leftslice.shape == rightslice.shape, "all images must be same size"
	
	leftpreds = np.reshape(leftpreds, (shape[0], shape[1], shape[2]))
	rightpreds = np.reshape(rightpreds, (shape[0], shape[1], shape[2]))
	leftslice = np.reshape(leftslice, (shape[0], shape[1], shape[2]))
	rightslice = np.reshape(rightslice, (shape[0], shape[1], shape[2]))

	for i in xrange(N):
		fig = plt.figure()	

		ax1 = fig.add_subplot(231)
		plt.imshow(leftslice[i],cmap='gray')
		plt.title('Actual left slice')
		plt.xticks([])
		plt.yticks([])
	
		ax2 = fig.add_subplot(232)
		plt.imshow(rightpreds[i],cmap='gray')
		plt.title('Predicted right slice')
		plt.xticks([])
		plt.yticks([])
	
		ax3 = fig.add_subplot(233)
		plt.imshow(rightslice[i],cmap='gray')
		plt.title('Actual right slice')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(234)
		plt.imshow(rightslice[i],cmap='gray')
		plt.title('Actual right slice')
		plt.xticks([])
		plt.yticks([])
		
		ax5 = fig.add_subplot(235)
		plt.imshow(leftpreds[i],cmap='gray')
		plt.title('Predicted left slice')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(236)
		plt.imshow(leftslice[i],cmap='gray')
		plt.title('Actual left slice')
		plt.xticks([])
		plt.yticks([])

		plt.tight_layout()
		plt.show(fig)


def plot_four_image_comparison(preds, rightslice, leftslice,N=10, reverse=False):
	shape = preds.shape
	preds = np.reshape(preds, (shape[0], shape[1], shape[2]))
	rightslice = np.reshape(rightslice,(shape[0], shape[1], shape[2]))
	leftslice = np.reshape(leftslice, (shape[0], shape[1], shape[2]))

	for i in xrange(N):
		fig = plt.figure()

		#originalcolour
		ax1 = fig.add_subplot(221)
		plt.imshow(leftslice[i])
		plt.title('Left slice')
		if reverse:
			plt.title('Right slice')
		plt.xticks([])
		plt.yticks([])

		#red
		ax2 = fig.add_subplot(222)
		plt.imshow(preds[i])
		plt.title('Predicted Right Slice')
		if reverse:
			plt.title('Predicted Left Slice')
		plt.xticks([])
		plt.yticks([])

		#green
		ax3 = fig.add_subplot(223)
		plt.imshow(leftslice[i])
		plt.title('Left slice')
		if reverse:
			plt.title('Right Slice')
		plt.xticks([])
		plt.yticks([])

		##blue
		ax4 = fig.add_subplot(224)
		plt.imshow(rightslice[i])
		plt.title('Actual Right slice')
		if reverse:
			plt.title('Actual Left Slice')
		plt.xticks([])
		plt.yticks([])

		plt.tight_layout()
		plt.show(fig)
		return fig



def get_both_preds_results(fname1, fname2):
	pass




if __name__ == '__main__':
	
	res = load_array("gestalt_half_split_results_proper")
	#res = load_array("STANDARD_WITH_GESTALT_AUTOENCODER_MODEL_1")
	history, predsleft, sliceleft, sliceright = res
	res2 = load_array("gestalt_half_split_results_proper_other")
	#res2 = load_array("STANDARD_WITH_GESTALT_AUTOENCODER_MODEL_2")
	history2, predsright, _,_2 = res2
	plot_both_six_image_comparison(predsleft, predsright, sliceleft, sliceright)
	
	
	
