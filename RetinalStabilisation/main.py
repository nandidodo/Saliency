# so this is where the main file is defined, and the results are hopefully gained
# it wll just be a erlatively simple mnist network, not particularly difficult at all
# and an mnist autoencoder, which seems reasonable
#and then it wll train networks on that and on the copied  dataset
# to see what happens there
# hopefully nothing too terrible!
# it might even work, and if it does, that will have been easy!

import numpy as np 
import keras
from augmenter import *
from utils import *
from models import *
from keras.datasets import mnist
import keras.backend as K
from keras import metrics

# aim will be to check if it works at all better - hopefully it will
# with the data augmentation
# and also eventually show the decreasing of prediction error
#with the copied images, so who knows!




# okay, let's define main function here, and build it up while testing
if __name__ == '__main__':
	NUM_AUGMENTS = 10
	PIXEL_SHIFT = 4
	BASE_SAVE_PATH = "data/mnist_dataset"
	#load the generated dataset
	augments = np.load(BASE_SAVE_PATH+'_augments')
	copies = np.load(BASE_SAVE_PATH + '_copies')
