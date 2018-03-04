# okay, so the aim here is thsi is where I write my experimental code for it
# to get the panorama image and stuff so it makes sense
# to see what realistic scanpaths will look like
# on a single image
#just testing
import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import cPickle as pickle
from panorama import *
from scanpaths import *
from utils import *

def plot_panorama_step(pan_img, viewport, sal_map, centre, viewport_width, viewport_height, sigma=None, cmap='gray', border_width=10):
	if sigma is not None:
		assert sigma>0,'Gaussian smooth factor sigma must be greater than 0'
		sal_map = gaussian_filer(sal_map, sigma)

	highlighted_img = highlight_viewport(pan_img, centre, viewport_width, viewport_height, border_width)
	#begin plot
	#show viewport, salience map, and then 
	fig = plt.figure()

	ax1 = fig.add_subplot(131)
	plt.imshow(viewport, cmap=cmap)
	plt.title('Input Image')
	plt.xticks([])
	plt.yticks([])
	
	ax2 = fig.add_subplot(132)
	plt.imshow(sal_map, cmap=cmap)
	plt.title('Salience map')
	plt.xticks([])
	plt.yticks([])

	ax3 = fig.add_subplot(133)
	plt.imshow(highlighted_img, cmap='gray')
	plt.title('Image within panorama')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	plt.show()



def test_panorama_scanpaths_single_image(pan_fname, model_fname, first_centre=None, viewport_width=100, viewport_height=100, N=20, show_results=True, sigma=None):

	#do asserts
	assert type(pan_fname)=='string' and len(pan_fname)>1, 'Panorama fname invalid'
	assert type(model_fname)==='string' and len(model_fname)>1, 'Panorama fname invalid'
	assert len(first_centre)==2, 'Image centre must be two dimensional'
	
	#load images
	pan_img = load_array(pan_fname)
	#reshape if necessary
	if len(pan_img).shape !=2:
		pan_img = attempt_image_reshape_2d(pan_img)
	#load keras model
	h,w = pan_img.shape
	try:
		model = load_model(model_fname)
	except:
		raise TypeError('Keras model could not be loaded with this filename')
	#check first centre is okay
	if first_centre=None:
		first_centre=(h//2, w[1]//2)
	ch,cw = first_centre
	#assert first centre is okay
	assert ch>=0 and ch<= h, 'Initial centre height must be within the panorama image'
	assert cw>=0 and cw<=w, 'Initial centre width must be within the panorama image'
	#check viewport widths and heights
	assert type(viewport_width)=='int', 'Viewport width must be an integer'
	assert type(viewport_height)=='int', 'Viewport height must be an integer'
	vw = viewport_width//2
	vh = viewport_height//2
	assert viewport_width<=w, 'Viewport width cannot be greater than the panorama image'
	assert viewport_height<=h, 'Viewport height cannot be greater than the image'

	#initialise	
	viewports = []
	sal_maps = []
	centres = []
	
	centre = first_centre
	centres.append(centre)
	#now begin the loop
	for i in xrange(N):
		#get the viewport
		viewport_img = move_viewport(panorama_img, centre, viewport_width, viewport_height)
		#get saliency map predictoins
		pred = mode.predict(viewport_img)
		#assume the pred is the sal map for now. Usually I average but I don't hav to do that here hopefully!
		#reset centre
		#if show
		if show_results:
			plot_panorama_step(pan_img, viewport_img, pred, centre viewport_width, viewport_height, sigma=sigma)
		
		centre = get_max_indices(pred)
		#now append
		viewports.append(viewport_img)
		sal_maps.append(pred)
		centres.append(centre)

	#now numpyify and return
	centres = np.array(centres)
	viewports= np.array(viewports)
	sal_maps = np.array(sal_maps)
	return viewports, sal_maps, centres

#First I'm going to need to train the model. For simplicity I've copied the function to do this here
# and it will also adapt it slightly

def plot_model_results(images, preds, N=10,cmap='gray'):
	assert len(images.shape)==3, 'Image shape must be two dimensional'
	assert len(preds.shape)==3,'Preds shape must be two dimensional'
	assert images.shape == preds.shape, 'Images and preds Dont have the same dimensions. There is probably a mismatch of some kind here'
	




def train_panorama_model_prototype(fname,epochs=100, both=True):
	imgs = load_array(fname)
	imgs = imgs.astype('float32')/255.
	#simply train on the green for ease
	imgs = imgs[:,:,:,0]
	shape = imgs.shape
	train,test= split_into_test_train(imgs)

	model = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
	model.compile(optimizer='sgd',loss='mse')
	callbacks = build_callbacks("./")
	his=model.fit(train, train, epochs=epochs, batch_size=128, shuffle=True, validation_data=(test,test), callbacks=callbacks)

	preds = model.predict(test)
	history = serialize_class_object(his)
	res = [history, preds, test]
	save_array(res, "PANORAMA_PROTOTYPE_MODEL_RESULTS")
	#save the model
	model.save("PANORAM_PROTOTYPE_MODEL")
	return res





			


## now test
if __name__ =='__main__':
	
	
	

