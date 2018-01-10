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

X = Input(shape=(n_x,))
label = Input(shape=(n_y,))
#now we concatenate x and y so we can igure this out
inputs = concat([X, label]) # not sure why we need this
# we need to merge them within the context of the graph

encoder_h = Dense(encoder_dim1, activation=activation, activity_regularizer='l2')(inputs)
mu = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h) # this is our mapping t omu and sigma via ANNs

#now we have a function which add random noise to our sampling process, and use a lambda layer in keras for this - cool!
def sample_z(args):
	mu, l_sigma = args
	eps = K.random_normal


