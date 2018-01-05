
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



def test_gestalt():
	imgs = load_array("testimages_combined")
	#print imgs.shape
	imgs = imgs[:,:,:,0].astype('float32')/255.
	shape = imgs.shape
	imgs = np.reshape(imgs, (shape[0], shape[1], shape[2], 1))
	train, test = split_into_test_train(imgs)
	slicelefttrain, slicerighttrain = split_dataset_center_slice(train, 20)
	slicelefttest, slicerighttest = split_dataset_center_slice(test,20)
	shape = slicelefttrain.shape
	
	#sort out our model
	model = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
	model.compile(optimizer='sgd', loss='mse')
	callbacks = build_callbacks("gestalt/")
	model.fit(slicelefttrain, slicerighttrain, epochs=5, batch_size=128, shuffle=True, validation_data=(slicelefttest, slicerighttest), callbacks=callbacks)



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


	model.fit(xtrain, xtrain, nb_epoch=5, batch_size=128, shuffle=True, validation_data=(xtest, xtest), verbose=1, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


if __name__ =='__main__':
	#test_cifar()
	test_gestalt()
