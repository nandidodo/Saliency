
# okay, so the previou sautoencoder model wasn't really working (at all!) so my lpan, I think,  is to try to make a gestalt autoencoder model which hopefully will work significantly better than before, and see if it does. let's hope it does. we can copy some of the VGG conv layers and stuff if it seems to be doing oay, and isn't too horrendously slow!

# so, what is the plan for tomorrow re this (5/1/18 (!!!)) argh it's 2018!!! anyhoow I think what I've got to do is basically just play around with the models and get a good one working and try using the inspiration of the vgg net orany other good cnn keras implementations that i can find on the web and get the gestalts working, and then possibly back apply them to the thing that we actually want for the main model. that would be very useful to have done tomorrow, and then in the afternoon of course it's the yelp clone and then eventually, hopefully enyo!!!

# okay, so how do we define models for things in keras and get them working? 
# this should be a fairly straightfoward thing to do to be honest?
# let's see if we can get it working in cifar and see if it works... could be nice?

# s yeah, I guess this isn't original at all but the point of the autoencoder is to find some function on the input which can be easily defined and larned in an unsupervised manner which requires using full informatino about the stucture of the iamge to represent, so things like denoising and so forth add useful little things there as do the cross prediction channel thing. it's basically the same as a denoising autoencoder but without any serious problems at all, and it's not that fun really, but could be seriosuly useful. the zhang paper is really cool and useful, and Ishould look up where they do that to see if I can get it to work atall, and it is interesting. let's see if it can kind of manage to learn gestalt kind of continuations as that should be really interesting from a psychological perspective which is really the point

import keras
import numpy as np
from gestalt_models import *
from utils import *
from experiments import *
from gestalt import *
import matplotlib.pyplot as plt


def split_first_test_train(data, frac_train = 0.9):
	assert frac_train <=1, "frac_train must be a fraction"
	frac_test = 1-frac_train
	N = len(data)
	train = data[int(frac_test*N):N]
	test = data[0:int(frac_test*N)]
	return train, test

def split_into_test_train(data, frac_train = 0.9, frac_test = 0.1):
	assert frac_train + frac_test == 1, 'fractions must add up to one'
	length = len(data)
	#print length
	#print frac_train*length
	train = data[0:int(frac_train*length)]
	test = data[int(frac_train*length): length]
	return train, test


def split_first_test_val_train(data, frac_train =0.9, frac_val = 0.05, frac_test = 0.05):
	assert frac_train + frac_val + frac_test ==1, "train test validation splits must add up to one"
	N = len(data)
	len_test = int(frac_test*N)
	len_val = int(frac_val*N)
	#len_train = int(frac_train *N)
	test = data[0:len_test]
	val = data[len_test: (len_test+len_val)]
	train =data[(len_test+len_val):N]
	return train, val, test 


def plot_both_six_image_comparison(leftpreds, rightpreds, leftslice, rightslice, N=10):
	shape = leftpreds.shape
	assert shape == rightpreds.shape == leftslice.shape == rightslice.shape, "all images must be same size"
	
	leftpreds = np.reshape(leftpreds, (shape[0], shape[1], shape[2]))
	rightpreds = np.reshape(rightpreds, (shape[0], shape[1], shape[2]))
	leftslice = np.reshape(leftslice, (shape[0], shape[1], shape[2]))
	rightslice = np.reshape(rightslice, (shape[0], shape[1], shape[2]))

	for i in xrange(N):
		fig = plt.figure()	
		fig = plt.figure()

		ax1 = fig.add_subplot(231)
		plt.imshow(leftslice[i])
		plt.title('Actual left slice')
		plt.xticks([])
		plt.yticks([])
	
		ax2 = fig.add_subplot(232)
		plt.imshow(rightpreds[i])
		plt.title('Predicted right slice')
		plt.xticks([])
		plt.yticks([])
	
		ax3 = fig.add_subplot(233)
		plt.imshow(rightslice[i])
		plt.title('Actual right slice')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(234)
		plt.imshow(rightslice[i])
		plt.title('Actual right slice')
		plt.xticks([])
		plt.yticks([])
		
		ax5 = fig.add_subplot(235)
		plt.imshow(leftpreds[i])
		plt.title('Predicted left slice')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(236)
		plt.imshow(leftslice[i])
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




def test_gestalt(both=False,epochs=500, fname="gestalt/default_gestalt_test"):
	imgs = load_array("testimages_combined")
	#print imgs.shape
	imgs = imgs[:,:,:,0].astype('float32')/255.
	shape = imgs.shape
	imgs = np.reshape(imgs, (shape[0], shape[1], shape[2], 1))
	#train, test = split_first_test_train(imgs)
	train, val, test = split_first_test_val_train(imgs)
	slicelefttrain, slicerighttrain = split_dataset_center_slice(train, 20)
	slicelefttest, slicerighttest = split_dataset_center_slice(test, 20)
	sliceleftval, slicerightval = split_dataset_center_slice(val, 20)
	#slicerighttest = split_dataset_center_slice(test,20)
	shape = slicelefttrain.shape

	print "SHAPES OF INPUTS:"
	print slicelefttrain.shape
	print slicerighttrain.shape
	print slicelefttest.shape
	print slicerighttest.shape
	
	#sort out our model
	model = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
	model.compile(optimizer='sgd', loss='mse')
	callbacks = build_callbacks("gestalt/")
	his = model.fit(slicelefttrain, slicerighttrain, epochs=500, batch_size=128, shuffle=True, validation_data=(sliceleftval, slicerightval), callbacks=callbacks)

	if both:
		model2 = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
		model2.compile(optimizer='sgd', loss='mse')
		his2 = model2.fit(slicerighttrain, slicelefttrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(sliceleftval, slicerightval), callbacks=callbacks)

	print "MODEL FITTED"

	preds = model.predict(slicelefttest)
	print preds.shape
	"""for i in xrange(10):
		plt.imshow(np.reshape(slicerighttest[i],(100,20)),cmap='gray')
		plt.title('image')
		plt.show()
		plt.imshow(np.reshape(preds[i],(100,20)), cmap='gray')
		plt.title('prediction')
		plt.show()
	"""
	history = serialize_class_object(his)
	res = [history,preds, slicelefttest, slicerighttest]
	save_array(res, fname+ "_1")

	#plot_four_image_comparison(preds, slicelefttest, slicerighttest, 20)

	if both:
		preds = model.predict(slicerighttest)
		print preds.shape
		"""for i in xrange(10):
			plt.imshow(np.reshape(slicerighttest[i],(100,20)),cmap='gray')
			plt.title('image')
			plt.show()
			plt.imshow(np.reshape(preds[i],(100,20)), cmap='gray')
			plt.title('prediction')
			plt.show()
		"""
		history = serialize_class_object(his2)
		res = [history,preds, slicelefttest, slicerighttest]
		save_array(res, fname + "_2")

		#plot_four_image_comparison(preds, slicerighttest, slicelefttest, 20)
	

	#okay, that's weird. it seems to learn to predict the right slice, even though it's not supposed to, and I have no idea whyy it's trying to do that, so I really don't know...
# ah, les, I have an idea actually. let's flip_this aroud and see if we get anything on the other side

# what's crazy is that this actually seems to work!?!??! that's totally insane and I've no idea how it does it. It'll definitely be something to show richard. We should also experiment with seeing how good the autoencoder is on the standard colour transfer task to see if we get any interesting errmaps. That's what I'll do tonight, I think
	

def test_cifar():
	(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
	xtrain = xtrain[:,:,:,0].astype('float32')/255.
	xtest = xtest[:,:,:,0].astype('float32')/255.
	xtrain = np.reshape(xtrain, (len(xtrain), 32,32,1))
	xtest = np.reshape(xtest, (len(xtest), 32,32,1))
	print xtrain.shape
	#model = SimpleAutoencoder((28,28,1))
	model=SimpleConvDropoutBatchNorm((32,32,1))
	model.compile(optimizer='sgd', loss='mse')


	model.fit(xtrain, xtrain, nb_epoch=500, batch_size=128, shuffle=True, validation_data=(xtest, xtest), verbose=1, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
	

# it actually seems to have worked really well!!! our model is really niec and good! that's awesome! next steps are getting more images, getting gestalt images, telling richard about it, and seeing what he says, and experimenting with different settings but the basic hyperparams seem to work really well this time, which is great!

if __name__ =='__main__':
	#test_cifar()
	test_gestalt(both=True, epochs=1)
	"""
	imgs = load_array('testsaliences_combined')
	imgs = imgs[:,:,:,0]
	print imgs.shape
	train, val, test = split_first_test_val_train(imgs)
	print train.shape
	print val.shape
	print test.shape
	"""
