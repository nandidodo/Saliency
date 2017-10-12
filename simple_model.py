# okay, this is where we do the simplest model available to me to hopefulyl help out with this, and ideally it would help a lot. so we just try to do the conv nets, and stuff, and make it decent with keras and the like, but still, we would still most definitely require the thing. most likely we would need a class definition and have methods on that, which seem reasonable,  but that's beyond me at this point, so we stick with simple scripting files till now! as, tbh, tht's what python is, a scripting language

from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def plot_sample_results(N = 10, start = 0):
	plt.figure(figsize=(20,4))
	for i in range(N):
		#display original
		ax = plt.subplot(2,N,i+1)
		plt.imshow(xtest[start + i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		#display reconstructoin
		ax = plt.subplot(2, N, i+1+N)
		plt.imshow(decoded_imgs[start + i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	
	plt.show()

def generate_error_maps(inputs, predictions):
	return np.absolute(inputs-predictions)

def plot_error_map(error_map):
	plt.imshow(error_map)
	plt.show()

def generate_salience_trace(error_map, N = 255):
	salience_arr = np.zeros(error_map.shape)
	for i in xrange(N):
		

def split_dataset_by_colour(data):
	red = data[:,:,:,0]
	blue = data[:,:,:,1]
	green = data[:,:,:,2]
	return [red, blue, green]

def split_img_by_colour(img):
	red = img[:,:,0]
	blue = img[:,:,1]
	green = img[:,:,2]
	return [red, blue, green]

#load our data
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
K.set_image_dim_ordering('th')

#randomised seed
seed = 8
np.random.seed(seed)

#casts and normalisations tothe data - very simple processing

# we do casts and normalisations
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain/255.0
xtest = xtest/255.0 # 0 to make sure it's a floating point division

#one hot encode outputs
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)
num_classes = ytest.shape[1]

#model params
dropout_rate = 0.3
conv_kernel = (3,3)
pool_kernel = (2,2)
activation = 'relu'

#training params
lrate = 0.01
epochs = 5
batch_size = 64
shuffle = True


input_img = Input(shape=(28,28,1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the encoder is finished, and it's time to move on to the decoder

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# now we define our model andcombine
autoencoder = Model(input_img, decoded)
# we should experiment with different optimisers and stuff. that's fairly standard
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# we train
autoencoder.fit(xtrain, xtrain, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_data=(xtest, xtest), callbacks =[TensorBoard(log_dir='tmp/autoencoder')])

# once trained we then move onto getting our predictions
preds = autoencoder.predict(xtest)

# we then try to compare them to the thing






	
