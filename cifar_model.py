# okay, this is the one where I finally define my model for cifar and also try to figure out the cross channel encoding with the split brain autoencoder model. If I can get this done, then that will be sufficient for tonight, I think. There is currently going to be large amounts of code duplicatoin, and this likely cannot be helped atm. I'll definitely have to refactor this all later, but for the moment we're just doing prototyping, so it's okay



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



def plot_sample_results(inputs, preds, N = 10, start = 0):
	plt.figure(figsize=(20,4))
	for i in range(N):
		#display original
		ax = plt.subplot(2,N,i+1)
		plt.imshow(inputs[start + i].reshape(32,32))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		#display reconstructoin
		ax = plt.subplot(2, N, i+1+N)
		plt.imshow(preds[start + i].reshape(32,32))
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
		plot_error_map(np.reshape(error_maps[start+i,:,:,:], [32,32]))

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


def salience_trace_with_gaussians(error_map, N = 5, std=4):
	#first we get our gaussians
	gauss_means = []
	for i in range(N,0,-1):
		map_index = np.argmax(error_map)
		error_map[map_index] = 0
		gauss_means.append(map_index)

	sal_map = np.zeros(error_map.shape)
	
	for a in xrange(len(error_map[:,1]):
		for b in xrange(len(error_map[1,:]):
			for c in xrange(len(gauss_means)):
				sal_map[a][b] += (1.0/index_distance([a,b], gauss_means[c]))* np.abs(np.random.normal(0, std))


	#this i sa realy hacky and horrible way of doing it, but it might work! to produce something vaguelly gaussianly smoothed!
	return sal_mal
		

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

#set image ordering
#K.set_image_dim_ordering('th')

#load our data
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

#randomised seed
seed = 8
np.random.seed(seed)

#casts and normalisations tothe data - very simple processing

# we do casts and normalisations
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain/255.0
xtest = xtest/255.0 # 0 to make sure it's a floating point division

print xtrain.shape
print xtest.shape

# reshape for mnist
#xtrain= np.reshape(xtrain, (len(xtrain), 3,32,32))
#xtest= np.reshape(xtest, (len(xtest), 3,32,32))

#split into red green blue
redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
redtest, greentest, bluetest = split_dataset_by_colour(xtest)

print redtrain.shape
print redtest.shape
redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
redtest = np.reshape(redtest, (len(redtest), 32,32,1))
greentest = np.reshape(greentest, (len(greentest), 32,32,1))
bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))


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
epochs = 20
batch_size = 64
shuffle = True



# okay, our convolutional autoencoder for stuff
print "MODEL SHAPES: "
print "  "
input_img = Input(shape=(32,32,1))
print input_img.shape
print "  "

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
print x.shape
#x = MaxPooling2D((2, 2), padding='same')(x)
#print x.shape
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print x.shape
#x = MaxPooling2D((2, 2), padding='same')(x)
#print x.shape
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print x.shape
encoded = MaxPooling2D((2, 2), padding='same')(x)
print encoded.shape
print "  "

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print x.shape
x = UpSampling2D((2, 2))(x)
print x.shape
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print x.shape
#x = UpSampling2D((2, 2))(x)
#print x.shape
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
print x.shape
#x = UpSampling2D((2, 2))(x)
#print x.shape
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
print decoded.shape
print "  "
print redtrain.shape

# like I mean this isn't functoinal because we're whacked out thing wrong, not probably having anything to do with the split brin idea, but smiply because it's just so bad. although to be perfeclty honest for all we know it could just not work. I honestly don't know. I could leave this running overnight to see if we get anything good. it doesn't seem - TOO deathly, but I don't know. I had to mess aroud really badly with the dimensions to get it to work, and I dno't really understand how the dimensions work, so it's probably me doing something horrible to the images which makes it impossible to learn vs anything in particular, but we can try!

# now we define our model andcombine
autoencoder = Model(input_img, decoded)
optimizer = optimizers.SGD(lr = lrate, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
# we train
autoencoder.fit(redtrain, redtrain, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_data=(redtest, redtest), callbacks =[TensorBoard(log_dir='tmp/autoencoder')])


# once trained we then move onto getting our predictions
preds = autoencoder.predict(redtest)

55.9648924  maps:52:9
-3.1866092

# we see our sample results
print "  "
print "results: greentest, preds:"
print greentest.shape
print preds.shape
plot_sample_results(redtest, preds)

# we generate the error maps here
error_maps = generate_error_maps(redtest, preds)
print error_maps.shape

# a quick save here
fname = "cifar_error_map_preliminary_no_split"
save(error_maps, fname)
fname2 = "cifar_predictions_preliminary_no_split"
save(preds, fname2)
#fname3 = "cifar_greentest"
#save(preds, greentest)

#print error_maps[1,:,:,:].shape
#print error_maps[1].shape

# we lpot some error maps
plot_error_maps(error_maps)

# we then generate some saliency maps off them and have a look
#show_salience_traces(error_maps)
