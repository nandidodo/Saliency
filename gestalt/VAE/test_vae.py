
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
from utils import *

img_rows, img_cols, img_chns = 28, 28, 1
if K.image_data_format() == 'channels_first':
		original_img_size = (img_chns, img_rows, img_cols)
else:
		original_img_size = (img_rows, img_cols, img_chns)

epochs = 1
batch_size = 100
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

	
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
activation = 'relu'


def vae_model(input_shape,epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std, activation='relu'):
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
		            strides=1)(conv_3)
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
	decoder_upsample = Dense(filters * rows/2 * cols/2, activation=activation)

	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, rows/2, cols/2)
	else:
		output_shape = (batch_size, rows/2, cols/2, filters)

	decoder_reshape = Reshape(output_shape[1:])
	decoder_deconv_1 = Conv2DTranspose(filters,
		                               kernel_size=num_conv,
		                               padding='same',
		                               strides=1,
		                               activation=activation)
	decoder_deconv_2 = Conv2DTranspose(filters,
		                               kernel_size=num_conv,
		                               padding='same',
		                               strides=1,
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
	xent_loss = reconstruction_loss(rows, cols, x, x_decoded_mean_squash)
	kl = kl_loss(z_mean, z_log_var)
	vae_loss = K.mean(xent_loss + kl)
	vae.add_loss(vae_loss)

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

	return vae, encoder, generator

	#okay, so at the moment the loss is utterly enormous. It's probably a problem with the loss function? but I'm not totally sure?

#split out the losses into functions. This seems to have worked so far!
def reconstruction_loss(rows, cols, x, x_decoded):
	return rows * cols * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded))

def kl_loss(z_mean, z_log_var):
	return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)



# train the VAE on MNIST digits
# so, let's try it on other images. for instance the benchmark image set, and so generalise it to take in any image size. If it still works then, then I'm doing well!
"""
(x_train, _), (x_test, y_test) = mnist.load_data()



x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
"""

imgs= load_array("testimages_combined")
imgs = imgs.astype('float32')/255. # normalise here. This might solve some issues
print(imgs.shape)
x_train, x_test = split_first_test_train(imgs)
print('x_train.shape:', x_train.shape)

vae, encoder, generator = vae_model(x_train.shape[1:],epochs, batch_size, filters, num_conv, latent_dim, intermediate_dim, epsilon_std)
vae.compile(optimizer='rmsprop',loss=None)
vae.summary()


vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))



# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
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

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
