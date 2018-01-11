#this is where I put my custom losses functions for the vaes
# and various other things to experiment with

import numpy as np
import keras
from keras import backend as K




def vae_loss(y_true, y_pred):
	recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
	kl= 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
	return recon + kl

class CustomVariationalLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder=True
		super(CustomVariationalLayer, self).__init__(**kwargs)

	#not sure what that does or means.. argh?

	def vae_loss(self, x, x_decoded_mean_squash):
		x = K.flatten(x)
		x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
		xent_loss = img_rows*img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
		kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):
		x = inputs[0]
		x_decoded_mean_squash = inputs[1]
		loss = self.vae_loss(x, x_decoded_mean_squash)
		self.add_loss(loss, inputs=inputs)
		return x

