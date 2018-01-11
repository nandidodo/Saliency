# okay, example keras vae built with keras from the tutorial - and includes a whole load of useful stuff
# so let's copy this andtry to udnerstand it
# then build our own model


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


# these are our hyperparams here briefly!
img_rows, img_cols, img_channels = 28,28,1
filters = 64
conv_kernel = 3
batch_size = 64
if (K.image_data_format() =='channels_first':
	original_img_size = (img_channels, img_rows, img_cols)
	intermediate_output_shape = (batch_size, filters, 14,14)
	output_shape=(batch_size, filters, 29,29)
else:
	original_img_size=(img_rows, img_cols, img_channels)
	intermediate_output_shape=(batch_size, 14, 14, filters) 
	output_shape = (batch_size, 29,29, filters)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
epochs = 10
activation = 'relu'

#now we define the model

#this is the encoder
x = Input(shape=original_img_size)
conv1 = Conv2D(img_channels, kernel_size=(2,2),padding='same', activation=activation)(x)
conv2 = Conv2D(filters, kernel_size=(2,2), padding='same', activation=activation), strides=(2,2))(conv1)
conv3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation=activation, strides=1)(conv2)
conv4 = Conv2D(filters, kernel_size=num_conv, padding='same',activation=activation, strides=1)(conv3)

#we flatten to preapre for the mapping to latent dims
flat = Flatten()(conv4)
hidden = Dense(intermediate_dim, activation=activation)(flat)

#now for the latent mapping
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

#reparametrisation trick
def sampling(args);
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=K.shape(z_mean)[0], latent_dim),
							mean=0., stdev=epsilon_std)
	return z_mean + K.exp(z_log_var) * epsilon


# now for latent space mapping layer
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# now we instantate these lauest separately to reuse them later
decoder_hid = Dense(intermediate_dim, activation=activation)
decoder_upsample=Dense(filters * 14 * 14, activation=activation)

decoder_reshape=Reshape(intermediate_output_shape[1:])
decoder_deconv1 = Conv2dTranspose(filters, kernel_size=num_conv, padding='same',strides=1, activation=activation)
decoder_deconv2 = Conv2DTranspose(flters, kernel_size=num_conv, padding='same', strides=1, activation=activation)
decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=(3,3), strides=(2,2), padding='valid', activation=activation)

decoder_mean_squash = Conv2D(img_channels, kernel_size=2, padding='valid', activation='sigmoid')

# not sure what this does. it's a very big model to be honest. we'll have to make our own separate package with utils and stuff and just copy it across see what works
#now we just utilize these layers and reuse them here for this stuff!
hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded=decoder_reshape(up_decoded)
deconv1_decoded=decoder_deconv1(reshape_decoded)
deconv2_decoded=decoder_deconv2(deconv1_decoded)
x_decoded_relu=decoder_deconv3(deconv2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# we define our custom variational loss layer here

class CustomVariationalLayer(Layer):
	def __init__(self, **kwargs):
		self.is_placeholder=True
		super(CustomVariationalLayer, self).__init__(**kwargs)

	#not sure what that does or means.. argh?

	def vae_loss(self, x, x_decoded_mean_squash):
		x = K.flatten(x)
		x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
		xent_loss = img_rows*img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
		kl_loss = -0.5 * K.mean(1 + z_log_var - K.sqaure(z_mean) - K.exp(z_log_var), axis=-1)
		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):
		x = inputs[0]
		x_decoded_mean_squash = inputs[1]
		loss = self.vae_loss(x, x_decoded_mean_squash)
		self.add_loss(loss, inputs=inputs)
		return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(x,y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

#train the vae on mnist
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = xtrain.astype('float32')/255.
xtrain = xtrain.reshape((xtrain.shape[0],) + original_img_size)
xtest = xtest.astype('float32')/255.
xtest = xtest.reshape((xtest.shape[0],)+original_img_size)

print (" xtrain shape: " + str(xtrain.shape)

vae.fit(xtrain, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(xtest, None))

#we build a model to project inputs on the latent space
encoder = Model(x, z_mean) # we only go to the z mean nad not the std for some reason

# we also build a generator
decoder_input = Input(shape=(latent_dim,))
_hid_decoded= decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded=decoder_reshape(_up_decoded)
_deconv1_decoded=decoder_deconv1(_reshape_decoded)
_deconv2_decoded=decoder_deconv2(_deconv1_decoded)
_x_deocded_relu=decoder_deconv_3_upsamp(_deconv2_decoded)
_x_decoded_mean_squash=decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# so we want to display a 2d plot of the digit classes in the latent space
x_test_encoded = encoder.predict(xtest, batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1],c=y_test)
plt.colorboar()
plt.show()

# and now we want to display a 2d manifold of the digits
n = 15 # we have a figure with 15x15 digits
digit_size  = 28
figure = np.zeros((digit_size*n, digit_size*n))
#linearly spaced coordinates on the unit square are transformed through inverse CDF of gaussian
# to produce values o fthe latent variables z sinec prior of latent space is gaussian
grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05, 0.95,n))

#I'm not really sure what ay of this does. hopefully it helps a bit thoguh!?
for i, yi in enumerate(grid_x):
	for j xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]])
		z_sample = np.tile(z_sample, batch_size).reshape(batch_size,2)
		x_decoded = generator.predict(z_sample, batch_size=batch_size)
		digit = x_decoded[0].reshape(digit_size, digit_size)
		figure[i*digit_size: (i+1) * digit_size,
				j*digit_size: (j+1)*digit_size] = digit
plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
