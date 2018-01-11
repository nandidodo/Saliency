#okay, so this is the actual convolutional conditional autoencoder which I'll try and write up
# before testing on the split half images to see if we can have any kind of training
# and see if it works. 
# but first we're going to test thatu p here

import warnings
import numpy as np

from keras_tqdm import TQDMNotebookCallback
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from scipy.misc import imsave
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

#import data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = xtrain.astype('float32')/255.
xtest = xtest.astype('float32')/255.

# we're going to only use dense nets so we'ev got to flatten the mnist things - seems bad really tbh
n_pixels = np.prod(xtrain.shape[1:])
xtrain = xtrain.reshape((len(xtrain), n_pixels))
xtest = xtest.reshape((len(xtest), n_pixels))

# we need to do labels - and also why are weusing these for an autoencoder??
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

# let's setup our hyperparams
m =  250 # batch_size!
n_z = 2 # latent space size - not big!!!
encoder_dim1 = 512
decoder_dim1 = 512
decoder_out_dim = 784
activation='relu'
optim = Adam(lr=0.0005)

n_x = xtrain.shape[1]
n_y = ytrain.shape[1]

epochs = 100

# okay, let's define our model

# okay, so the variational autoeencoder consists of two parts - firstly the encoder which is a mapping/NN representation of the probability distribution p(z|X) i.e. it maps from the input space X to the latent variable space Z. we want to maximise the usefulness ot this thing and also how closely it approximates the standard normal N(0,1) as regularisation
# secondly we want to map in the decoder from the latent variable to another X X^ and therefore maximise p(X|z) to match the data which is a stanard log thing, so I don't know
# we'll generally want a much more dimensional latent variable, and I'm not sure how I shuold get it tbh
# like the mu and sigma are just meant to be a standard normal vector, right? in multiple dimensions
# so we should be able to deal with that, but we can't, so what do we do there?

X = Input(shape=(n_x,))
label = Input(shape=(n_y,))
#now we concatenate x and y so we can igure this out
inputs = concat([X, label]) # not sure why we need this
# we need to merge them within the context of the graph

encoder_h = Dense(encoder_dim1, activation=activation, activity_regularizer='l2')(inputs)
mu = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h) # this is our mapping t omu and sigma via ANNs

#now we have a function which add random noise to our sampling process, and use a lambda layer in keras for this - cool!
# this is the reparametrization trick. basicaly, we're meant to be sampling from our calculated z normal N(mu, sigma) which are the outputs of the neural network, but this isn't differentiable, so what we do is instead sample from the standard normal, which isn't differentiable, but this doesn't matter as it's not actually dependent on anything in the network or trainable, so it doesn't need to ahve gradients, and then we multiply and add the other bits, which are differentiable, so the reparametrisation works, and the network is differentiable through these things

def sample_z(args):
	mu, l_sigma = args
	eps = K.random_normal
	return mu + K.exp(log_sigma/2) * eps

# we use a lambda layer to sample our standard thing
z = Lambda(sample_z)([mu, l_sigma])
	#
#now we build a very simple decoder network

decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(284, activation='sigmoid')
h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# so we now have three things we can do, reconstruct inputs, encoder inputs into latentvariables
# and generate data from latent variables. so let's look at this
vae = Model(inputs, outputs)
encoder = Model(inputs, mu) # we use mean ot the output as it'sthe center point, the representative of the gaussian

# and we can also have a generator ourputs
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

# we also need to have our custom vae loss combining both the KL and the reconstruction loss

def vae_loss(y_true, y_pred):
	recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
	kl= 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
	return recon + kl


# and then we train it
vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(xtrain, xtrain, batch_size=m, epochs=epochs)
