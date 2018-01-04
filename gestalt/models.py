# okay, this is where we write fairly simple gestalt models, with the aim of hopefully being simple
# so let's test a few

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def SimpleConv(weights_path=None):
	model = Sequential()
