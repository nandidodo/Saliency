# okay, this is where I actually run my experiments, as another kind of master file/script

from keras.datasets import cifar10, mnist
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras.models import Model
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from file_reader import *
from utils import *


def normalise(data):
	return data.astype('float32')/255.0

seed = 8
np.random.seed(seed)

# let's try something

(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

xtrain = normalise(xtrain)
xtest = normalise(xtest)

redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
redtest, grentest, bluetest = split_dataset_by_colour(xtest)

redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
redtest = np.reshape(redtest, (len(redtest), 32,32,1))
greentest = np.reshape(greentest, (len(greentest), 32,32,1))
bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

# okay, that sorts out our data, now let's get the model working

a1 = Hemisphere(redtrain, greentrain, redtest, greentest)
a2 = Hemisphere(greentrain, redtrain, greentest, redtest)

a1.train(epochs=1)
a2.train(epochs=1)

a1.plot_results()
a2.plot_results()

errmap1 = a1.get_error_maps()
errmap2 = a2.get_error_maps()

a1.plot_error_maps(errmap1)
a2.plot_error_maps(errmap2)

