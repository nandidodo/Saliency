
'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
# Reference
- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.models import load_model
from keras.datasets import cifar10
from utils import *

img_rows, img_cols, img_chns = 28, 28, 1
if K.image_data_format() == 'channels_first':
		original_img_size = (img_chns, img_rows, img_cols)
else:
		original_img_size = (img_rows, img_cols, img_chns)

epochs = 25
batch_size = 100
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

	
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
activation = 'relu'


def vae_model(input_shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std, activation='relu', save=True, save_fname="results/gestalt_VAE"):
	# input image dimensions
	rows, cols, channels = input_shape


	x = Input(shape=input_shape)
	conv_1 = Conv2D(img_chns,
		            kernel_size=(2, 2),
		            padding='same', activation=activation)(x)
	conv_2 = Conv2D(filters,
		            kernel_size=(2, 2),
		            padding='same', activation=activation,
		            strides=(2, 2))(conv_1)
	conv_3 = Conv2D(filters,
		            kernel_size=num_conv,
		            padding='same', activation=activation,
		            strides=1)(conv_2)
	conv_4 = Conv2D(filters,
		            kernel_size=num_conv,
		            padding='same', activation=activation,
		            strides=2)(conv_3)
	flat = Flatten()(conv_4)
	hidden = Dense(intermediate_dim, activation=activation)(flat)

	z_mean = Dense(latent_dim)(hidden)
	z_log_var = Dense(latent_dim)(hidden)


	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
		                          mean=0., stddev=epsilon_std)
		return z_mean + K.exp(z_log_var) * epsilon

	# note that "output_shape" isn't necessary with the TensorFlow backend
	# so you could write `Lambda(sampling)([z_mean, z_log_var])`
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	# we instantiate these layers separately so as to reuse them later
	decoder_hid = Dense(intermediate_dim, activation=activation)
	#decoder_upsample = Dense(filters * 14 * 14, activation=activation)
	decoder_upsample = Dense(filters * rows/4 * cols/4, activation=activation)

	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, rows/4, cols/4)
	else:
		output_shape = (batch_size, rows/4, cols/4, filters)

	decoder_reshape = Reshape(output_shape[1:])
	decoder_deconv_1 = Conv2DTranspose(filters,
		                               kernel_size=num_conv,
		                               padding='same',
		                               strides=1,
		                               activation=activation)
	decoder_deconv_2 = Conv2DTranspose(filters,
		                               kernel_size=num_conv,
		                               padding='same',
		                               strides=2,
		                               activation=activation)
	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, rows+1, cols+1)
	else:
		output_shape = (batch_size, rows+1, cols+1, filters)
	decoder_deconv_3_upsamp = Conv2DTranspose(filters,
		                                      kernel_size=(3, 3),
		                                      strides=(2, 2),
		                                      padding='valid',
		                                      activation=activation)
	decoder_mean_squash = Conv2D(channels,
		                         kernel_size=2,
		                         padding='valid',
		                         activation='sigmoid')

	hid_decoded = decoder_hid(z)
	up_decoded = decoder_upsample(hid_decoded)
	reshape_decoded = decoder_reshape(up_decoded)
	deconv_1_decoded = decoder_deconv_1(reshape_decoded)
	deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
	x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
	x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

	#y = Input(shape=input_shape)

	# instantiate VAE model
	vae = Model(x, x_decoded_mean_squash)

	# Compute VAE loss
	#xent_loss = img_rows * img_cols * metrics.binary_crossentropy(
	#	K.flatten(x),
	#	K.flatten(x_decoded_mean_squash))
	#kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	#this straight up isn't working. The loss is going crazy and I don't know why?
	# I literally haven't changed anything except the sizes of the images? I'm not sure why it's suddenly broken here. This is really annoying?
	# okay, so loss is decreasing. it's just enormous at the moment, and I don't know why!
	 # it seems very unstable and we almost certainly don't have enough image data to learn this properly... ah! I could obtain more first probably - via imagenet or something?
	#the loss is decreasing though. Who will know what will happen in the end?
	#xent_loss = reconstruction_loss(rows, cols, x, x_decoded_mean_squash)
	kl = kl_loss(z_mean, z_log_var)
	#vae_loss = K.mean(xent_loss + kl)
	vae.add_loss(kl)

	# build a model to project inputs on the latent space
	encoder = Model(x, z_mean)

	#generator
	# build a digit generator that can sample from the learned distribution
	decoder_input = Input(shape=(latent_dim,))
	_hid_decoded = decoder_hid(decoder_input)
	_up_decoded = decoder_upsample(_hid_decoded)
	_reshape_decoded = decoder_reshape(_up_decoded)
	_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	generator = Model(decoder_input, _x_decoded_mean_squash)

	#implement model saving simply here
	if save:
		vae.save(save_fname + "_VAE")
		encoder.save(save_fname+ "_encoder")
		generator.save(save_fname + "_generator")

	return vae, encoder, generator, z_mean, z_log_var

	#okay, so at the moment the loss is utterly enormous. It's probably a problem with the loss function? but I'm not totally sure?

#split out the losses into functions. This seems to have worked so far!
def reconstruction_loss(y, x_decoded):
	#let's hard code this for now
	rows = 28
	cols = 28
	rec_loss = rows * cols * metrics.binary_crossentropy(K.flatten(y), K.flatten(x_decoded))
	print("Rec loss: " + str(rec_loss))
	return rec_loss
def unnormalised_reconstruction_loss(x_decoded, y):
	rec_loss = metrics.binary_crossentropy(K.flatten(x_decoded), K.flatten(y))
	print("Rec loss: " + str(rec_loss))
	return rec_loss

def kl_loss(z_mean, z_log_var):
	klloss =  -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	print("KL loss: " + str(klloss))
	return klloss

# oh wow!!! this might be working!!!!! I guess only time will tell here?!!?!?!??!
# we have a reasonable lack of loss, which is good. but oh wow. If this works it could unlock the whole gestalt thing properly, which would be fantastic!

# train the VAE on MNIST digits
# so, let's try it on other images. for instance the benchmark image set, and so generalise it to take in any image size. If it still works then, then I'm doing well!

# okay, so I'm pretty sure my altered network works for mnist. The question then is that it simply does not scale to the larger images. Let's try to solve this initially by scaling down our images, and see if we can get interesting gestalt effects. It might be even more interesting just to try this out on mnist first, which we KNOW actually works! and then if we get interesting gestalt effects, try to get it on the proper images!

# it works for mnist. So now I need to try doing the splitting in half image to see if that works



def predict_display(N, testslices, actuals,generator):
	testsh = testslices.shape
	actualsh = actuals.shape
	#if len(testsh) ==3:
	testslices = np.reshape(testslices,(testsh[0], testsh[1], testsh[2]))
	actuals = np.reshape(actuals, (actualsh[0], actualsh[1], actualsh[2]))

	#epsilon_std = 1.0
	for i in xrange(N):
		#epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
		                          #mean=0., stddev=epsilon_std)
		#epsilon = np.random.multivariate_normal(z_mean, np.exp(z_log_var))
		#print("Z MEAN")
		##print(z_mean)
		#print("Z LOG VAR")
		#print(z_log_var)
		#z =  z_mean + K.exp(z_log_var) * epsilon
		#print(z)
		##try evaling it
		#z = K.eval(z)
		#print(z)
	#let's just do this with completely standard multivariate normal. see if there's decent results
		z = np.random.multivariate_normal([0,0],[[1,0],[0,1]])
		z = np.reshape(z, (1,2))
		pred = generator.predict(z,batch_size=1)
		sh = pred.shape
		pred = np.reshape(pred, (sh[1], sh[2],sh[3]))
		if sh[3] ==1:
			pred = np.reshape(pred, (sh[1],sh[2]))
		print("PRED")
		print(pred.shape)
		print("testslice")
		print(testslices[i].shape)
		print("actual")
		print(actuals[i].shape)
		plot_three_image_comparison(testslices[i], pred, actuals[i], reshape=False)

def mnist_experiment():
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train = x_train.astype('float32') / 255.
	x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
	x_test = x_test.astype('float32') / 255.
	x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

	lefttrain, righttrain = split_dataset_center_slice(x_train, 12)
	lefttest, righttest = split_dataset_center_slice(x_test, 12)
	#print(lefttrain.shape
	#imgs= load_array("testimages_combined")
	#imgs = imgs.astype('float32')/255. # normalise here. This might solve some issues
	#print(imgs.shape)
	#x_train, x_test = split_first_test_train(imgs)
	#print('x_train.shape:', x_train.shape)
	shape = lefttrain.shape[1:]
	#shape=lefttrain.shape[1:]

	vae, encoder, generator, z_mean, z_log_var = vae_model(shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std)
	vae.compile(optimizer='adam',loss=reconstruction_loss)
	vae.summary()

	callbacks = build_callbacks("results/callbacks/")

	his = vae.fit(lefttrain,righttrain,
		shuffle=True,epochs=epochs, batch_size=batch_size,
		validation_data=(lefttest, righttest))

	#okay, yay!!! it works really really really wel with all of mnist. That's great. Now I need to try to apply it to cifar... see if it can be done in the slightest. One would hope that it could!!

	#okay, so this totally fails even on the simplest and least demanding of mnist tasks
	#I am entirely uncertain why it does so or what is going wrong? I guess the only thing to do is perhaps start again from the tutorial example and do more rigorous quality control at each step. dagnabbit. this is just really frustrating.
	#maybe the trouble is that the kl loss massively outweights the reconstruction lsos
	#so I just end up with a thing. let's try it with the normaliesd reconstruction loss!

	#okay, this works now! that's pretty great!
	#just for quick tests of the thing
	#x_test = lefttest
	#let's try it now with the split data, to see if there are any interesting results

	history = serialize_class_object(his)
	save_array(history, "results/VAE_train_history_2")
	#save models
	vae.save('results/VAE_vae_model_1')
	generator.save('results/VAE_generator_model_1')
	encoder.save('results/VAE_encoder_model_1')


	# display a 2D plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(lefttest, batch_size=batch_size)
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
	#plt.colorbar()
	plt.show()

	# build a digit generator that can sample from the learned distribution
	#decoder_input = Input(shape=(latent_dim,))
	#_hid_decoded = decoder_hid(decoder_input)
	#_up_decoded = decoder_upsample(_hid_decoded)
	#_reshape_decoded = decoder_reshape(_up_decoded)
	#_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	#_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	#_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	#_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	#generator = Model(decoder_input, _x_decoded_mean_squash)

	preds = vae.predict(lefttest)
	save_array(preds, "results/mnist_vae_preds_2")
			
	predict_display(20, lefttest, x_test, generator)
	##Tensor("add_1:0", shape=(?, 2), dtype=float32)


	# display a 2D manifold of the digits
	n = 15  # figure with 15x15 digits
	digit_width = shape[0]
	digit_height = shape[1]
	figure = np.zeros((digit_width * n, digit_height * n))
	# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
	# to produce values of the latent variables z, since the prior of the latent space is Gaussian
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
		    z_sample = np.array([[xi, yi]])
		    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
		   # print(z_sample)
		    x_decoded = generator.predict(z_sample, batch_size=batch_size)
		    digit = x_decoded[0,:,:,0].reshape(digit_width, digit_height)
		    figure[i * digit_width: (i + 1) * digit_width,
		           j * digit_height: (j + 1) * digit_height] = digit

	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()


def cifar10_experiment():
	# so cifar isn't workingat all. but mnist does, if I'm not mistaken. We've got to figure out therefore ,what is wrong with the CIFAR code
	slice_width = 16
	epochs=20
	(xtrain, ytrain),(xtest, ytest) = cifar10.load_data()
	xtrain = xtrain.astype('float32')/255.
	xtest = xtest.astype('float32')/255.
	sh = xtrain.shape
	#let's reshape to only be 2d like mnist. that could be causing the issues
	# yes! that's working a little better. except no. it's just getting stuck at 0.6933 vs mnist. I'm not sure why this is the case, but it's infuriating as it's not working at all
	#could be an issue with the learning rate?
	xtrain = np.reshape(xtrain[:,:,:,0],(len(xtrain), sh[1], sh[2],1))
	xtest = np.reshape(xtest[:,:,:,0], (len(xtest), sh[1],sh[2],1))

	lefttrain, righttrain = split_dataset_center_slice(xtrain, slice_width)
	lefttest, righttest = split_dataset_center_slice(xtest, slice_width)
	#print(lefttrain.shape
	#imgs= load_array("testimages_combined")
	#imgs = imgs.astype('float32')/255. # normalise here. This might solve some issues
	#print(imgs.shape)
	#x_train, x_test = split_first_test_train(imgs)
	#print('x_train.shape:', x_train.shape)
	shape = lefttrain.shape[1:]
	#shape=lefttrain.shape[1:]

	vae, encoder, generator, z_mean, z_log_var = vae_model(shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std)
	vae.compile(optimizer='adam',loss=reconstruction_loss)
	vae.summary()

	callbacks = build_callbacks("results/callbacks/")

	his = vae.fit(lefttrain,righttrain,
		shuffle=True,epochs=epochs, batch_size=batch_size,
		validation_data=(lefttest, righttest))

	#okay, yay!!! it works really really really wel with all of mnist. That's great. Now I need to try to apply it to cifar... see if it can be done in the slightest. One would hope that it could!!

	#okay, so this totally fails even on the simplest and least demanding of mnist tasks
	#I am entirely uncertain why it does so or what is going wrong? I guess the only thing to do is perhaps start again from the tutorial example and do more rigorous quality control at each step. dagnabbit. this is just really frustrating.
	#maybe the trouble is that the kl loss massively outweights the reconstruction lsos
	#so I just end up with a thing. let's try it with the normaliesd reconstruction loss!

	#okay, this works now! that's pretty great!
	#just for quick tests of the thing
	#x_test = lefttest
	#let's try it now with the split data, to see if there are any interesting results

	history = serialize_class_object(his)
	save_array(history, "results/VAE_train_history_cifar_1")
	#save models
	vae.save('results/VAE_vae_model_1_cifar')
	generator.save('results/VAE_generator_model_1_cifar')
	encoder.save('results/VAE_encoder_model_1_cifar')


	# display a 2D plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(lefttest, batch_size=batch_size)
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
	#plt.colorbar()
	plt.show()

	# build a digit generator that can sample from the learned distribution
	#decoder_input = Input(shape=(latent_dim,))
	#_hid_decoded = decoder_hid(decoder_input)
	#_up_decoded = decoder_upsample(_hid_decoded)
	#_reshape_decoded = decoder_reshape(_up_decoded)
	#_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	#_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	#_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	#_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	#generator = Model(decoder_input, _x_decoded_mean_squash)

	preds = vae.predict(lefttest)
	save_array(preds, "results/cifar_vae_preds_1")
			
	predict_display(20, lefttest, xtest, generator)
	##Tensor("add_1:0", shape=(?, 2), dtype=float32)


	# display a 2D manifold of the digits
	n = 15  # figure with 15x15 digits
	digit_width = shape[0]
	digit_height = shape[1]
	figure = np.zeros((digit_width * n, digit_height * n))
	# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
	# to produce values of the latent variables z, since the prior of the latent space is Gaussian
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
		    z_sample = np.array([[xi, yi]])
		    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
		   # print(z_sample)
		    x_decoded = generator.predict(z_sample, batch_size=batch_size)
		    digit = x_decoded[0,:,:,0].reshape(digit_width, digit_height)
		    figure[i * digit_width: (i + 1) * digit_width,
		           j * digit_height: (j + 1) * digit_height] = digit

	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()



	"""
	print(xtrain.shape)
	lefttrain, righttrain = split_dataset_center_slice(xtrain, slice_width)
	lefttest, righttest = split_dataset_center_slice(xtest, slice_width)
	print(lefttrain.shape)
	#let's print them to make sure we're doing okay
	
	for i in xrange(20):
		fig = plt.figure()
		print(lefttrain[i].shape)
		l = np.reshape(lefttrain[i], (32,12))
		r = np.reshape(righttrain[i], (32,12))
		ax1 = fig.add_subplot(121)
		plt.imshow(l)
		ax2 = fig.add_subplot(122)
		plt.imshow(r)
		plt.show(fig)
	
	
	shape = lefttrain.shape[1:]

	vae, encoder, generator, z_mean, z_log_var = vae_model(shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std,save_fname="results/vae_cifar_model")
	vae.compile(optimizer='sgd',loss=unnormalised_reconstruction_loss)
	vae.summary()

	callbacks = build_callbacks("results/callbacks/")

	his = vae.fit(lefttrain,lefttrain,
		shuffle=True,epochs=epochs, batch_size=batch_size,
		validation_data=(lefttest, righttest))

	#just for quick tests of the thing
	x_test = lefttest

	history = serialize_class_object(his)
	save_array(history, "results/VAE_train_history_cifar")


	# display a 2D plot of the digit classes in the latent space
	x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
	#plt.colorbar()
	plt.show()

	preds = vae.predict(x_test)
	save_array(preds, "results/cifar_vae_preds")
			
	predict_display(20, lefttest, x_test, generator)
	#okay, well then that's great. This doesn't work at all. even in the best of circumstances
	#I don't understand why.. dagnabbit!? It was woring before!
	"""


# another thing we could work on is active inference in the standard sense in rl environments. first we need to get a working framework to work in though, so that's really the challenge of this week

# I think I know why the other gestalt network was not working... we didn't actually have it learning the ys, so they did nothing. That was useful


# the next step is figuring out how to train this in a reasonable manner so it works with things
# instead of just decoding itself, and comparing with itself

if __name__ == '__main__':
	cifar10_experiment()
	#mnist_experiment()
	#first thing to do is to check that it works absolutely. Not totally sure.
	#second thing in the standard one is to check the loss func works directly as appropriate?
	#I'm not sure how to implement it perfectly though, so I should look at this!
	#right so it learns absolutely nothing with the cifar. tomorrow's job is figuring out why

