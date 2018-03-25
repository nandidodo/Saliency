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
# first I should just check out the overall error there, as far as I know

def run_mnist_model(train, test,save_name=None,epochs=100, Model=SimpleConvDropoutBatchNorm, batch_size = 128, save_model_name=None):
	#normalise data
	train = train.astype('float32')/255.
	test = test.astype('float32')/255.

	#get the model
	shape = train.shape
	model = Model(shape[1:])
	#for now for simplicity
	model.compile(optimizer='sgd', loss='mse')
	#callbacks = build_callbacks('/callbacks')
	his = model.fit(train, train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test, test))
	history = serialize_class_object(his)

	#get predictions
	preds = model.predict(test)

	#save the results
	if save_model_name is not None:
		model.save(save_model_name)

	#save the preds and test and history
	res = [test, preds, his]
	if save_name is not None:
		save_array(res,save_name)

	return res





# okay, let's define main function here, and build it up while testing
if __name__ == '__main__':
	NUM_AUGMENTS = 10
	PIXEL_SHIFT = 4
	BASE_SAVE_PATH = "data/mnist_dataset"
	EPOCHS = 1
	BATCH_SIZE = 64
	#load the generated dataset
	augments_train = np.load(BASE_SAVE_PATH+'_train_augments')
	copies_train = np.load(BASE_SAVE_PATH + '_train_copies')
	augments_test = np.load(BASE_SAVE_PATH+'_train_augments')
	copies_test = np.load(BASE_SAVE_PATH + '_train_copies')

	#augments results
	run_mnist_model(augments_train, augments_test, save_name="mnist_augments", epochs=EPOCHS, batch_size=BATCH_SIZE,save_model_name="model_mnist_augments")

	#copy results
	run_mnist_model(copies_train, copies_test, save_name="mnist_copies", epochs=EPOCHS, batch_size=BATCH_SIZE,save_model_name="model_mnist_copy")