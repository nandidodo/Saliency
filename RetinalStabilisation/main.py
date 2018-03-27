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
import gc

# aim will be to check if it works at all better - hopefully it will
# with the data augmentation
# and also eventually show the decreasing of prediction error
#with the copied images, so who knows!
# first I should just check out the overall error there, as far as I know

def run_mnist_model(train, test,save_name=None,epochs=100, Model=SimpleConvDropoutBatchNorm, batch_size = 128, save_model_name=None, ret = False):
	#normalise data
	train = train.astype('float32')/255.
	test = test.astype('float32')/255.
	#reshape for input to conv network
	shape = train.shape
	train = np.reshape(train, (len(train), shape[1], shape[2],1))
	test = np.reshape(test, (len(test), shape[1], shape[2],1))
	shape = train.shape

	#get the model

	model = Model(shape[1:])
	#for now for simplicity
	model.compile(optimizer='sgd', loss='mse')
	#callbacks = build_callbacks('/callbacks')
	print "loaded data and compiled model"
	his = model.fit(train, train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test, test))
	#train is no longer needed, so free it
	print "fitted model"
	train = 0
	print "freed train"
	history = serialize_class_object(his)
	his = 0
	print "loaded history and freed his"

	#get predictions
	preds = model.predict(test)
	print "made predictions"

	#save the results
	if save_model_name is not None:
		model.save(save_model_name)
		print "saved model"

	#save the preds and test and history
	if save_name is not None:
		#save_array(res,save_name)
		#tis is the issue. this takes up too much memory
		# it's because during te pickle dump, it requires the duplication
		# of theo bject in memory, which exploded my computer
		# so you REALLY can't uespickle for large arrays at all!
		# it was the res which was probably taking up so much space,
		#being copied twice!
		# and is otherwise copletely dire
		# and the pickle results don't work
		# so instead I'm just going to save stuff separately
		np.save(save_name+'_preds', preds)
		save_array(history, save_name+'_history')
		np.save(save_name+'_test', test)
		print "save results"

	if ret:
		print "returning"
		return [test, preds, history]

	res=None
	preds = None
	history=None
	print "freed variables"

	return


def run_augments():
	augments_train = np.load(BASE_SAVE_PATH+'_train_augments.npy')
	#augments results
	augments_test = np.load(BASE_SAVE_PATH+'_train_augments.npy')
	print "Loaded augments"
	run_mnist_model(augments_train, augments_test, save_name="mnist_augments", epochs=EPOCHS, batch_size=BATCH_SIZE,save_model_name="model_mnist_augments")
	print "finished running model"
	#"free" the memory by reassigning once it's no longer needed
	augments_train = 0
	augments_test = 0
	print "Freed data"
	return

def run_copies():
	
	copies_train = np.load(BASE_SAVE_PATH + '_train_copies.npy')
	copies_test = np.load(BASE_SAVE_PATH + '_train_copies.npy')
	#copy results
	run_mnist_model(copies_train, copies_test, save_name="mnist_copies", epochs=EPOCHS, batch_size=BATCH_SIZE,save_model_name="model_mnist_copy")

	#"free" memory again
	copies_train = 0
	copies_test = 0 
	return




# okay, let's define main function here, and build it up while testing
if __name__ == '__main__':
	NUM_AUGMENTS = 10
	PIXEL_SHIFT = 4
	BASE_SAVE_PATH = "data/mnist_dataset"
	EPOCHS = 1
	BATCH_SIZE = 64
	#load the generated dataset
	
	# this should stop it blowing up my computer hopefully!

	#okay, this still blows up everything. I'm going to try putting it in separate functions now
	#hopefully since functions clean up after themselves this will stop the memory
	#exploding here

	run_augments()
	#also force garbage collection after each
	gc.collect()

	#run_copies()
	#gc.collect()

