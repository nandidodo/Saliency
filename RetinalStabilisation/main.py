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
from augmenter import *
from keras.models import load_model


# aim will be to check if it works at all better - hopefully it will
# with the data augmentation
# and also eventually show the decreasing of prediction error
#with the copied images, so who knows!
# first I should just check out the overall error there, as far as I know

def run_mnist_model(train, test,save_name=None,epochs=100, Model=SimpleConvDropoutBatchNorm, batch_size = 128, save_model_name=None, ret = False):
	#normalise data
	print "starting mnist model"
	train = train.astype('float32')/255.
	test = test.astype('float32')/255.
	#reshape for input to conv network
	shape = train.shape
	train = np.reshape(train, (len(train), shape[1], shape[2],1))
	test = np.reshape(test, (len(test), shape[1], shape[2],1))
	shape = train.shape
	print "train shape: ", train.shape
	print "test shape: " , test.shape

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

def mnist_discriminative(train, test, train_labels,test_labels save_name=None, epochs=10, Model=SimpleConvolutionalDiscriminative,batch_size = 128, save_model_name=None, ret = False):
	
	assert epochs>=1, 'Epochs must be at least one'
	assert batch_size>=1, 'Batch size must be at least 1'

	if type(train) is str:
		train = np.load(train)
	if type(test) is str:
		test = np.load(test)
	if type(train_labels) is str:
		train_labels = np.load(train_labels)
	if type(test_labels) is str:
		test_labels = np.load(test_labels)

	if save_name is not None:
		assert type(save_name) is str and len(save_name)>0, 'Save name must be a valid string'
	if save_model_name is not None:
		assert type(save_model_name) is str and len(save_model_name)>0, 'Save name must be a valid string'

	N_train = len(train)
	N_test = len(test)
	assert N_train == len(train_labels), 'Length of training dataset and labels must be the same'
	assert N_test == len(test_labels),'Length f test dataset and labels must be the same'

	#asserts complete, begin the actual trining
	#first begin by reshaping and normalising
	train = train.astype('float32')/255.
	test = test.astype('float32')/255.
	shape = train.shape
	train = np.reshape(train, (N_train, shape[1], shape[2], 1))
	test = np.reshape(test, (N_test, shape[1], shape[2], 1))
	print "train shape: ", train.shape
	print "test shape: " , test.shape


	model = Model(shape[1:])
	#for now for simplicity
	model.compile(optimizer='sgd', loss='mse')
	#callbacks = build_callbacks('/callbacks')
	print "loaded data and compiled model"
	his = model.fit(train, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test, test_labels))
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

def fixation_simulation(img, num_augments,run_num,epochs, copy_model_save=None, augment_model_save=None, results_save=None):

	# this runs for both the copies and the miages data
	# to get the results just ine one function

	#start models from scratch if save is none
	if copy_model_save is not None:
		assert augment_model_save is not None, 'Both models must be present or not'
	if augment_model_save is not None:
		assert copy_model_save is not None, 'Both models must be present or not'


	if copy_model_save is None and augment_model_save is None:
		Model=SimpleConvDropoutBatchNorm
		shape = img.shape
		shape=(1,shape[0], shape[1], 1)
		print shape
		copy_model = Model(shape[1:])
		copy_model.compile(optimizer='sgd', loss='mse')
		augment_model = Model(shape[1:])
		augment_model.compile(optimizer='sgd', loss='mse')
		augment_results = []
		copy_results = []

	if copy_model_save is not None and augment_model_save is not None:
		copy_model = load_model(copy_model_save)
		augment_model = load_model(augment_model_save)
		augment_results = []
		copy_results = []

	for i in xrange(run_num):
		augment_data = augment_with_translations(img, num_augments)
		copy_data = augment_with_copy(img, num_augments)

		#reshape data
		sh = augment_data.shape
		assert sh == copy_data.shape, 'Augment and copy data should have same shape'
		augment_data = np.reshape(augment_data, (sh[0], sh[1], sh[2],1))
		copy_data = np.reshape(copy_data, (sh[0], sh[1], sh[2],1))

		#fit the models
		augment_history = augment_model.fit(augment_data, augment_data, epochs=epochs, batch_size=1, shuffle=True)
		copy_history = copy_model.fit(copy_data, copy_data, epochs=epochs, batch_size=1, shuffle=True)

		#make predictions
		augment_preds = augment_model.predict(augment_data)
		copy_preds = copy_model.predict(copy_data)
		#augment_preds = np.reshape(augment_preds, (11,28,28))
		#copy_preds = np.reshape(copy_preds, (11,28,28))
		#plt.imshow(augment_preds[0])
		#plt.show()
		#plt.imshow(copy_preds[0])
		#plt.show()
		#augment_preds = np.reshape(augment_preds, (11,28,28,1))
		#copy_preds = np.reshape(copy_preds, (11,28,28,1))
		# this should work because the model should be the same each time
		# the model is preserved hopefully
		# if not then I can rerun each tiem and it should not matter
		# and save it each time, so to check if it does not work this time!
		augment_errmaps = get_error_maps(augment_data, augment_preds)
		copy_errmaps = get_error_maps(copy_data, copy_preds)
		#plt.imshow(augment_errmaps)
		#plt.show()
		#plt.imshow(copy_errmaps)
		#plt.show()
		# I mean I don't need to save the errmaps since htey can always be rederived
		# but it doesn't hurt and makes analysis easier and I've no shortage of space really!
		augment_results.append([augment_data, augment_preds, augment_errmaps])
		copy_results.append([copy_data, copy_preds, copy_errmaps])


	#save
	if results_save is not None:
		save(augment_results,results_save+'_fixation_augments')
		save(copy_results,results_save+'_fixation_copy')
	return augment_results, copy_results
	




def run_augments():
	augments_train = np.load(BASE_SAVE_PATH+'_train_augments.npy')
	#augments results
	augments_test = np.load(BASE_SAVE_PATH+'_test_augments.npy')
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
	copies_test = np.load(BASE_SAVE_PATH + '_test_copies.npy')
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
	EPOCHS = 10
	BATCH_SIZE = 64
	#load the generated dataset
	
	# this should stop it blowing up my computer hopefully!

	#okay, this still blows up everything. I'm going to try putting it in separate functions now
	#hopefully since functions clean up after themselves this will stop the memory
	#exploding here

	#run_augments()
	#also force garbage collection after each
	#gc.collect()

	#run_copies()
	#gc.collect()


	#run the fixation experiments now!
	(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
	num_augments = 10
	run_num = 1000
	epochs = 1
	copy_model_save = 'model_mnist_copy'
	augment_model_save = 'model_mnist_augments'
	results_save = 'results/from_scratch'
	#fixation_simulation(xtest[0], num_augments, run_num, epochs, copy_model_save, augment_model_save, results_save)
	fixation_simulation(xtest[0], num_augments, run_num, epochs, results_save=results_save)
	#the fixation results seem promising
	# so hopefully the results will be what I want
	# then the only thing to do will perhaps to do the classification
	# experiments and then write it up, which shall commence next week
	# and improve plotsandso forth, then have two papers written/done in two weeks
	# which is not too bad a rate. then I can spend some more time on larger projects
	# hopefully while still having a good phd showing overall
	# by figuring out what to do with the previous work on gestalt stuff or whatever
	# and also working on the predictive processing model of everything. which should be the main goal
	# for the rest of the month to be honest, and therefore working on the pred net
	# and the framework forthat, which seems to be rather important overall since it would mark
	# a real and moderately exciting contribution to thinsg which would be cool
	# that's april's overall goal then, I think ,and possibly a reasonable achievable one!
	# yay!
	# and also probably write up the first year review, which will be annoying and which
	# I seriously need to talk to richard about... dagnabbit
	# but first these papers. and then I'll be fine! and also the predictive processing stuff
	# so I really don't know about that at all, so hey, who knows?


