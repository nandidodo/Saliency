# okay, this is where we do the simplest model available to me to hopefulyl help out with this, and ideally it would help a lot. so we just try to do the conv nets, and stuff, and make it decent with keras and the like, but still, we would still most definitely require the thing. most likely we would need a class definition and have methods on that, which seem reasonable,  but that's beyond me at this point, so we stick with simple scripting files till now! as, tbh, tht's what python is, a scripting language

from keras.datasets import cifar10, mnist
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras.models import Model
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from file_reader import *
from utils import *


def plot_sample_results(inputs, preds, N = 10, start = 0):
	plt.figure(figsize=(20,4))
	for i in range(N):
		#display original
		ax = plt.subplot(2,N,i+1)
		plt.imshow(inputs[start + i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		#display reconstructoin
		ax = plt.subplot(2, N, i+1+N)
		plt.imshow(preds[start + i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	
	plt.show()

def generate_error_maps(inputs, predictions):
	return np.absolute(inputs-predictions)

def plot_error_map(error_map):
	plt.imshow(error_map)
	plt.show()

def plot_error_maps(error_maps, N= 10, start = 0):
	for i in xrange(N):
		#we get the index and reshape all in one!
		plot_error_map(np.reshape(error_maps[start+i,:,:,:], [28,28]))

def generate_salience_trace(error_map, N = 255):
	salience_arr = np.zeros(error_map.shape)
	# we loop backwards because we want the first values to be the highest
	for i in range(N, 0, -1):
		#we get the max index
		max_index = np.argmax(error_map)
		#we set that point on the error map as 0
		error_map[max_index] = 0
		# we set the new point on the salience map as the value
		salience_arr[max_index] = N
	
	return salience_arr
		

def show_salience_traces(error_maps, N=10, start = 0):
	for i in xrange(N):
		plt.imshow(generate_salience_trace(error_maps[i]))
		plt.show()


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
#(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

# we actually import mnist here for test purposes
(xtrain, _), (xtest, _) = mnist.load_data()
#K.set_image_dim_ordering('th')

#randomised seed
seed = 8
np.random.seed(seed)

#casts and normalisations tothe data - very simple processing

# we do casts and normalisations
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain/255.0
xtest = xtest/255.0 # 0 to make sure it's a floating point division

# reshape for mnist
xtrain= np.reshape(xtrain, (len(xtrain), 28,28,1))
xtest= np.reshape(xtest, (len(xtest), 28,28,1))

#one hot encode outputs
#ytrain = np_utils.to_categorical(ytrain)
#ytest = np_utils.to_categorical(ytest)
#num_classes = ytest.shape[1]

#model params
dropout_rate = 0.3
conv_kernel = (3,3)
pool_kernel = (2,2)
activation = 'relu'

#training params
lrate = 0.01
epochs = 10
batch_size = 64
shuffle = True



# okay, our convolutional autoencoder for stuff
input_img = Input(shape=(28,28,1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# now we define our model andcombine
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# we train
autoencoder.fit(xtrain, xtrain, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_data=(xtest, xtest), callbacks =[TensorBoard(log_dir='tmp/autoencoder')])


# once trained we then move onto getting our predictions
preds = autoencoder.predict(xtest)

# we see our sample results
plot_sample_results(xtest, preds)

# we generate the error maps here
error_maps = generate_error_maps(xtest, preds)
print error_maps.shape

# a quick save here
##fname = "error_map_test"
#save(error_maps, fname)

#print error_maps[1,:,:,:].shape
#print error_maps[1].shape

# we lpot some error maps
plot_error_maps(error_maps)

# we then generate some saliency maps off them and have a look
#show_salience_traces(error_maps)






	
