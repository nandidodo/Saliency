
# okay, so the previou sautoencoder model wasn't really working (at all!) so my lpan, I think,  is to try to make a gestalt autoencoder model which hopefully will work significantly better than before, and see if it does. let's hope it does. we can copy some of the VGG conv layers and stuff if it seems to be doing oay, and isn't too horrendously slow!

# so, what is the plan for tomorrow re this (5/1/18 (!!!)) argh it's 2018!!! anyhoow I think what I've got to do is basically just play around with the models and get a good one working and try using the inspiration of the vgg net orany other good cnn keras implementations that i can find on the web and get the gestalts working, and then possibly back apply them to the thing that we actually want for the main model. that would be very useful to have done tomorrow, and then in the afternoon of course it's the yelp clone and then eventually, hopefully enyo!!!

# okay, so how do we define models for things in keras and get them working? 
# this should be a fairly straightfoward thing to do to be honest?
# let's see if we can get it working in cifar and see if it works... could be nice?

# s yeah, I guess this isn't original at all but the point of the autoencoder is to find some function on the input which can be easily defined and larned in an unsupervised manner which requires using full informatino about the stucture of the iamge to represent, so things like denoising and so forth add useful little things there as do the cross prediction channel thing. it's basically the same as a denoising autoencoder but without any serious problems at all, and it's not that fun really, but could be seriosuly useful. the zhang paper is really cool and useful, and Ishould look up where they do that to see if I can get it to work atall, and it is interesting. let's see if it can kind of manage to learn gestalt kind of continuations as that should be really interesting from a psychological perspective which is really the point

# first we've got to run the test gestalt single model experiment and find those resulst. that's important. We'll run them over night, then tomorrow the autoencoder vae on mnist to check it works, and then if it does, run that too and hope for the best there, see if we get anything. also focus on GANs. That info MCMC thing looks vital! also info VAE study that.

# yeah, we need tocheck results of this, see if it works better. and then run the autoencoder on mnist, and then what? I really don't nkow how it works. I put too much code up online, and everything is just going wrong today and I'm donig everything wrong and I hate it and ugh.

# what are the steps we need to udnerstand for this to function correctly? Because we are far far behind literally everyone in the world and we need to be decent. We need to train this, to train the rejection sampler, and do some tests and experiments on the vaes. all of that is important. now let's run the autoencoder train

# okay, we do need to think substantially about efficiency and testing and stuff that is really useful, and how to produce efficient code rather than most github commits as that stuff doesn't really matter although it feels good. what really matters in the long term for me is progress, and we need to figureo ut how to do it explicitly, so let's try!

import keras
import numpy as np
from gestalt_models import *
from utils import *
from experiments import *
from gestalt import *
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras import metrics


def split_first_test_train(data, frac_train = 0.9):
	assert frac_train <=1, "frac_train must be a fraction"
	frac_test = 1-frac_train
	N = len(data)
	train = data[int(frac_test*N):N]
	test = data[0:int(frac_test*N)]
	return train, test

def split_into_test_train(data, frac_train = 0.9, frac_test = 0.1):
	assert frac_train + frac_test == 1, 'fractions must add up to one'
	length = len(data)
	#print length
	#print frac_train*length
	train = data[0:int(frac_train*length)]
	test = data[int(frac_train*length): length]
	return train, test


def split_first_test_val_train(data, frac_train =0.9, frac_val = 0.05, frac_test = 0.05):
	assert frac_train + frac_val + frac_test ==1, "train test validation splits must add up to one"
	N = len(data)
	len_test = int(frac_test*N)
	len_val = int(frac_val*N)
	#len_train = int(frac_train *N)
	test = data[0:len_test]
	val = data[len_test: (len_test+len_val)]
	train =data[(len_test+len_val):N]
	return train, val, test 


def plot_both_six_image_comparison(leftpreds, rightpreds, leftslice, rightslice, N=10):
	shape = leftpreds.shape
	#assert shape == rightpreds.shape == leftslice.shape == rightslice.shape, "all images must be same size"
	
	leftpreds = np.reshape(leftpreds, (leftpreds.shape[0], leftpreds.shape[1], leftpreds.shape[2]))
	rightpreds = np.reshape(rightpreds, (rightpreds.shape[0], rightpreds.shape[1], rightpreds.shape[2]))
	leftslice = np.reshape(leftslice, (leftslice.shape[0], leftslice.shape[1], leftslice.shape[2]))
	rightslice = np.reshape(rightslice, (rightslice.shape[0], rightslice.shape[1], rightslice.shape[2]))

	for i in xrange(N):
		fig = plt.figure()	

		ax1 = fig.add_subplot(231)
		plt.imshow(leftslice[i])
		plt.title('Actual left slice')
		plt.xticks([])
		plt.yticks([])
	
		ax2 = fig.add_subplot(232)
		plt.imshow(rightpreds[i])
		plt.title('Predicted right slice')
		plt.xticks([])
		plt.yticks([])
	
		ax3 = fig.add_subplot(233)
		plt.imshow(rightslice[i])
		plt.title('Actual right slice')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(234)
		plt.imshow(rightslice[i])
		plt.title('Actual right slice')
		plt.xticks([])
		plt.yticks([])
		
		ax5 = fig.add_subplot(235)
		plt.imshow(leftpreds[i])
		plt.title('Predicted left slice')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(236)
		plt.imshow(leftslice[i])
		plt.title('Actual left slice')
		plt.xticks([])
		plt.yticks([])

		plt.tight_layout()
		plt.show(fig)


def plot_four_image_comparison(preds, rightslice, leftslice,N=10, reverse=False):
	shape = preds.shape
	preds = np.reshape(preds, (shape[0], shape[1], shape[2]))
	rightslice = np.reshape(rightslice,(shape[0], shape[1], shape[2]))
	leftslice = np.reshape(leftslice, (shape[0], shape[1], shape[2]))

	for i in xrange(N):
		fig = plt.figure()

		#originalcolour
		ax1 = fig.add_subplot(221)
		plt.imshow(leftslice[i])
		plt.title('Left slice')
		if reverse:
			plt.title('Right slice')
		plt.xticks([])
		plt.yticks([])

		#red
		ax2 = fig.add_subplot(222)
		plt.imshow(preds[i])
		plt.title('Predicted Right Slice')
		if reverse:
			plt.title('Predicted Left Slice')
		plt.xticks([])
		plt.yticks([])

		#green
		ax3 = fig.add_subplot(223)
		plt.imshow(leftslice[i])
		plt.title('Left slice')
		if reverse:
			plt.title('Right Slice')
		plt.xticks([])
		plt.yticks([])

		##blue
		ax4 = fig.add_subplot(224)
		plt.imshow(rightslice[i])
		plt.title('Actual Right slice')
		if reverse:
			plt.title('Actual Left Slice')
		plt.xticks([])
		plt.yticks([])

		plt.tight_layout()
		plt.show(fig)
		return fig

def plot_three_image_comparison(slices, predicted_slices,other_slices,N=20):
	shape = slices.shape
	preds = np.reshape(predicted_slices, (shape[0], shape[1], shape[2]))
	rightslice = np.reshape(other_slices,(shape[0], shape[1], shape[2]))
	leftslice = np.reshape(slices, (shape[0], shape[1], shape[2]))
	for i in xrange(N):
		print "in three image cmoparison loop"
		fig = plt.figure()

		#originalcolour
		ax1 = fig.add_subplot(131)
		plt.imshow(leftslice[i],cmap='gray')
		plt.title('Input Slice')
		plt.xticks([])
		plt.yticks([])

		#red
		ax2 = fig.add_subplot(132)
		plt.imshow(preds[i],cmap='gray')
		plt.title('Predicted Other Slice')
		plt.xticks([])
		plt.yticks([])

		#green
		ax3 = fig.add_subplot(133)
		plt.imshow(rightslice[i],cmap='gray')
		plt.title('Actual Other Slice')
		plt.xticks([])
		plt.yticks([])

		
		plt.tight_layout()
		plt.show(fig)
		#return fig


def test_gestalt_single_model(epochs=500, fname="gestalt/single_model_test_32px", Model=SimpleSequentialModel, save_model=True, save_model_fname="gestalt/default_single_model_32px", loss_func = 'mse',data_fname="testimages_combined"):
	print "IN FUCNTION"
	imgs = load_array(data_fname)
	imgs = imgs[:,:,:,0].astype('float32')/255.
	shape = imgs.shape
	imgs = np.reshape(imgs, (shape[0],shape[1],shape[2],1))
	train, val,test = split_first_test_val_train(imgs)
	slicelefttrain, slicerighttrain = split_dataset_center_slice(train, 32)
	slicelefttest, slicerighttest = split_dataset_center_slice(test, 32)
	sliceleftval, slicerightval = split_dataset_center_slice(val, 32)

	#now we concat these things together
	print slicelefttrain.shape
	half1train = np.concatenate((slicelefttrain, slicerighttrain), axis=0)
	half2train = np.concatenate((slicerighttrain, slicelefttrain), axis=0)
	
	half1val = np.concatenate((sliceleftval, slicerightval), axis=0)
	half2val = np.concatenate((slicerightval, sliceleftval), axis=0)
	
	half1test = np.concatenate((slicelefttest, slicerighttest), axis=0)
	half2test = np.concatenate((slicerighttest, slicelefttest), axis=0)
	print half1train.shape
	print half1test.shape
	shape = half1train.shape

	model = Model((shape[1], shape[2], shape[3])) # is this a cause of bugs here... there shuoldn't be a shape[3], right
	model.compile(optimizer='sgd', loss=loss_func)
	callbacks = build_callbacks("gestalt/")
	his = model.fit(x=half1train, y=half2train, epochs=epochs, batch_size=128, shuffle=True, validation_data=(half1val, half2val), callbacks=callbacks)
	history = serialize_class_object(his)
	print model
	print type(model)
	preds1 = model.predict(half1test)
	preds2 = model.predict(half2test)

	if save_model:
		model.save(save_model_fname)

	res=[preds1, preds2, history, half1test, half2test]
	save_array(res, fname)

	#benchmark predictions

	benchmark_imgs = load_array("datasets/Benchmark/BenchmarkDATA/BenchmarkIMAGES_images_resized_100x100")
	benchmark_imgs = benchmark_imgs.astype('float32')/255.
	benchmark_imgs = benchmark_imgs[:,:,:,0]
	sh = benchmark_imgs.shape
	print benchmark_imgs.shape
	benchmark_imgs = np.reshape(benchmark_imgs, (sh[0], sh[1],sh[2],1))
	leftslice, rightslice = split_dataset_center_slice(benchmark_imgs, 32)
	benchmark_test = np.concatenate((leftslice, rightslice), axis=0)
	benchmark_preds = model.predict(benchmark_test)
	save_array(benchmark_preds, "gestalt/Benchmark_single_model_test_set_prediction_32px")
	
	return [model, preds1, preds2, history, benchmark_preds]


def test_loss_func(x,y):
	print "IN loss function"
	#print x.shape
	##print y.shape
	#x = K.eval(x)
	#y = K.eval(y)
	#print "post eval!"
	#print x.shape
	#print y.shape
	#print type(x)
	#print type(y)
	
	#print x	
	#print y

	#compare_two_images(x,y)
	#the above are sanity checks seeing if the correct images are actually fed into the loss functoin
	#and now I'll just return an utterly standard binary cross-entropy loss functio
	return metrics.binary_crossentropy(K.flatten(x), K.flatten(y))

def test_gestalt(both=False,epochs=500, fname="gestalt/default_gestalt_test", Model=SimpleConvDropoutBatchNorm, save_model=True, save_model_fname="gestalt/default_gestalt_model", loss_func = 'mse'):
	# has model function for additional generality here, which is great!
	imgs = load_array("testimages_combined")
	#print imgs.shape
	imgs = imgs[:,:,:,0].astype('float32')/255.
	shape = imgs.shape
	imgs = np.reshape(imgs, (shape[ 
0], shape[1], shape[2], 1))
	#train, test = split_first_test_train(imgs)
	train, val, test = split_first_test_val_train(imgs)
	slicelefttrain, slicerighttrain = split_dataset_center_slice(train, 20)
	slicelefttest, slicerighttest = split_dataset_center_slice(test, 20)
	sliceleftval, slicerightval = split_dataset_center_slice(val, 20)
	#slicerighttest = split_dataset_center_slice(test,20)
	shape = slicelefttrain.shape

	#plot_both_six_image_comparison(slicelefttrain, sliceleftval, slicelefttest,slicelefttrain)
	"""
	for i in xrange(10):
		fig = plt.figure()
		sh = slicelefttrain.shape
		ax = plt.subplot(121)
		
		plt.imshow(np.reshape(slicelefttrain[i],(sh[1],sh[2])))
		
		ax2 = plt.subplot(122)
		plt.imshow(np.reshape(slicerighttrain[i],(sh[1],sh[2])))
	
		plt.tight_layout()
		plt.show(fig)

	"""

	print "SHAPES OF INPUTS:"
	print slicelefttrain.shape
	print slicerighttrain.shape
	print slicelefttest.shape
	print slicerighttest.shape
	
	#sort out our model
	model = Model((shape[1], shape[2], shape[3]))
	model.compile(optimizer='sgd', loss=loss_func)
	callbacks = build_callbacks("gestalt/")
	his = model.fit(slicelefttrain, slicerighttrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(sliceleftval, slicerightval), callbacks=callbacks)

	if both:
		model2 = Model((shape[1], shape[2], shape[3]))
		model2.compile(optimizer='sgd', loss=loss_func)
		his2 = model2.fit(slicerighttrain, slicelefttrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(sliceleftval, slicerightval), callbacks=callbacks)

	print "MODEL FITTED"

	preds = model.predict(slicelefttest)
	print preds.shape
	"""for i in xrange(10):
		plt.imshow(np.reshape(slicerighttest[i],(100,20)),cmap='gray')
		plt.title('image')
		plt.show()
		plt.imshow(np.reshape(preds[i],(100,20)), cmap='gray')
		plt.title('prediction')
		plt.show()
	"""
	history = serialize_class_object(his)
	res = [history,preds, slicelefttest, slicerighttest]
	save_array(res, fname+ "_1")

	#plot_four_image_comparison(preds, slicelefttest, slicerighttest, 20)

	if both:
		preds = model.predict(slicerighttest)
		print preds.shape
		"""for i in xrange(10):
			plt.imshow(np.reshape(slicerighttest[i],(100,20)),cmap='gray')
			plt.title('image')
			plt.show()
			plt.imshow(np.reshape(preds[i],(100,20)), cmap='gray')
			plt.title('prediction')
			plt.show()
		"""
		history = serialize_class_object(his2)
		res = [history,preds, slicelefttest, slicerighttest]
		save_array(res, fname + "_2")

	if save_model:
		model.save(save_model_fname + "_1")
		model2.save(save_model_fname + "_2")


	#this will test our model with completely unseen data, which is very important
	#and larger data than it expects!?!?!? which might mess it up actually
	#I guess we'll find out!
	#but this is to tell if it's actually any good at what it does at all, or if it isn't
	#and it's somehow cheating by training on the validatoin set or whatever
	#I guess we'll have to see
	#if this works it's prtty hard proof since it's most definitely never seen any of these images
	 # before... so I guess we'll just have to look and see!
	imgs = load_array("datasets/Benchmark/BenchmarkDATA/BenchmarkIMAGES_images_resized_100x100")
	imgs = imgs.astype('float32')/255.
	print imgs.shape
	imgs = imgs[:,:,:,0]
	imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
	sliceleft, sliceright = split_dataset_center_slice(imgs, 20)
	rightpreds = model.predict(sliceleft)
	leftpreds = model2.predict(sliceright)
	res = [leftpreds, rightpreds]
	save_array(res, "gestalt/BCE_BenchmarkDataTestSetGestaltPrediction")

		#plot_four_image_comparison(preds, slicerighttest, slicelefttest, 20)
	

	#okay, that's weird. it seems to learn to predict the right slice, even though it's not supposed to, and I have no idea whyy it's trying to do that, so I really don't know...
# ah, les, I have an idea actually. let's flip_this aroud and see if we get anything on the other side

# what's crazy is that this actually seems to work!?!??! that's totally insane and I've no idea how it does it. It'll definitely be something to show richard. We should also experiment with seeing how good the autoencoder is on the standard colour transfer task to see if we get any interesting errmaps. That's what I'll do tonight, I think

# okay, this is really really weird.  have no idea how it actually manages to do it.. .argh!?
#like how can it know??? those are test images it's showing... wtf?? how does it manage to know that... argh?


def test_cifar():
	(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
	xtrain = xtrain[:,:,:,0].astype('float32')/255.
	xtest = xtest[:,:,:,0].astype('float32')/255.
	xtrain = np.reshape(xtrain, (len(xtrain), 32,32,1))
	xtest = np.reshape(xtest, (len(xtest), 32,32,1))
	slicelefttrain, slicerighttrain = split_dataset_center_slice(xtrain, 16)
	slicelefttest, slicerighttest= split_dataset_center_slice(xtest,16)
	print xtrain.shape
	#model = SimpleAutoencoder((28,28,1))
	
	model=SimpleConvDropoutBatchNorm((32,16,1))
	model.compile(optimizer='sgd', loss=test_loss_func)


	his = model.fit(slicelefttrain, slicerighttrain, nb_epoch=25, batch_size=128, shuffle=True, validation_data=(slicelefttest, slicerighttest), verbose=1, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
	history = serialize_class_object(his)
	preds = model.predict(slicelefttest)
	save_array(preds, 'gestalt/TEST_CIFAR_PREDS_2')
	plot_four_image_comparison(preds, slicelefttest, slicerighttest, 20)
	#okay, let's see if this simple cifar test works at all!
	#okay, our experiments are running.  Let's get some of this sorted!

# okay, lets do the simplest thing possible = go back to mnist!!
def test_mnist():
	(x_train, _), (x_test, y_test) = mnist.load_data()
	x_train = x_train.astype('float32') / 255.
	x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
	x_test = x_test.astype('float32') / 255.
	x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

	lefttrain, righttrain = split_dataset_center_slice(x_train, 12)
	lefttest, righttest = split_dataset_center_slice(x_test, 12)

	model=SimpleConvDropoutBatchNorm((28,12,1))
	model.compile(optimizer='sgd', loss=test_loss_func)


	his = model.fit(slicelefttrain, slicerighttrain, epochs=1, batch_size=128, shuffle=True, validation_data=(slicelefttest, slicerighttest), verbose=1, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
	history = serialize_class_object(his)
	preds = model.predict(slicelefttest)
	save_array(preds, 'gestalt/TEST_MNIST_PREDS_3')
	plot_four_image_comparison(preds, slicelefttest, slicerighttest, 20)
	

# it actually seems to have worked really well!!! our model is really niec and good! that's awesome! next steps are getting more images, getting gestalt images, telling richard about it, and seeing what he says, and experimenting with different settings but the basic hyperparams seem to work really well this time, which is great!

# okay, everthing with our model works, we can now test... yay!
#let's get this show on the road!

if __name__ =='__main__':
	#test_cifar()
	test_mnist()
	#test_gestalt(both=True, epochs=500, fname="BCE_gestalt_results",save_model_fname="gestalt/BCE_SimpleConvBatchNormModel", loss_func='binary_crossentropy')

	#test_gestalt_single_model(epochs=500)
	"""
	imgs = load_array('testsaliences_combined')
	imgs = imgs[:,:,:,0]
	print imgs.shape
	train, val, test = split_first_test_val_train(imgs)
	print train.shape
	print val.shape
	print test.shape
	

	imgs = load_array("datasets/Benchmark/BenchmarkDATA/BenchmarkIMAGES_images")
	print type(imgs)
	print len(imgs)
	print imgs.shape
	imgs = imgs[:,:,:,0]
	print imgs.shape
	plt.imshow(imgs[0])
	plt.show()
	"""

	"""
	preds = load_array("gestalt/BenchmarkDataTestSetGestaltPrediction")
	benchmark_imgs = load_array("datasets/Benchmark/BenchmarkDATA/BenchmarkIMAGES_images_resized_100x100")
	benchmark_imgs = benchmark_imgs.astype('float32')/255.
	benchmark_imgs = benchmark_imgs[:,:,:,0]
	sh = benchmark_imgs.shape
	print benchmark_imgs.shape
	benchmark_imgs = np.reshape(benchmark_imgs, (sh[0], sh[1],sh[2],1))
	leftslice, rightslice = split_dataset_center_slice(benchmark_imgs, 20)
	benchmark_test = np.concatenate((leftslice, rightslice), axis=0)
	benchmark_test2 = np.concatenate((rightslice, leftslice), axis=0)
	plot_three_image_comparison(benchmark_test, preds, benchmark_test2, N=100)
	"""

	"""	
	#let's try loading the model
	model = load_model("gestalt/default_single_model_32px")
	benchmark_imgs = load_array("datasets/Benchmark/BenchmarkDATA/BenchmarkIMAGES_images_resized_100x100")
	benchmark_imgs = benchmark_imgs.astype('float32')/255.
	benchmark_imgs = benchmark_imgs[:,:,:,0]
	sh = benchmark_imgs.shape
	benchmark_imgs = np.reshape(benchmark_imgs, (sh[0], sh[1],sh[2],1))
	leftslice, rightslice = split_dataset_center_slice(benchmark_imgs, 32)
	benchmark_test = np.concatenate((leftslice, rightslice), axis=0)
	sh = benchmark_test.shape
	for i in xrange(20):
		img = np.reshape(benchmark_test[i], (1, sh[1],sh[2],1))
		pred = model.predict(img)
		display_img = np.reshape(img, (sh[1],sh[2]))
		pred = np.reshape(pred, (sh[1],sh[2]))
		compare_two_images(pred, display_img)
	#preds1, preds2, history, half1test, half2test = load_array("gestalt/single_model_test")
	#plot_three_image_comparison(half1test, preds2, half2test)
	# okay, yeah, so as I suspect all it tries to do is just straight up predict the image it's given
	#as it can't realistically display the thing. and with the 32 it's not necessarily any better tbh - 
	# so yeah, this straight up doesn't work? because it just learns to predict it's own image
	# I'm kind of confused about what it's learning to do as it never learns totaly successfully
	#and usually the learning just straight up doesn't work. Let's try it with the vae's instead, as could be interesting! let's try to figure out the vae and get that working?
	"""































	
	#this also works too incredibly well, which I do not trust or understand what I'm doing wrong here. I'mvery sure the arrays are going in at the correct time and it's not just learning the images it's given, at least here it had no choice, it was predicting, and the graphing is in the right place too, I'm pretty sure so I literally do not understand how thi sworks! it's WAY WAY WAY WAY WAY too good at learning and understanding the images... dagnabbit!


	# okay, so this isrealy strange here!? this time it fails completely the wrong way around # but on the supposed test set it doesn't? I'm so confused. On this test set it just learns the image it's given, which also makes no sense to be perfectly honest, but on the benchmark image set it somehow doesn't do that. I'm just so utterly and totally confused here, and undoubtedly the dissapointing one is that it learns the iamge it's given, but that's still really strange to be honest in this test set, maybe it'sj ust traing in a really rubbish way?
	

# let's try different loss functions see if that helps - I think the kullback leibler wouldbe very interesting personally to see i fit works, so we'll try that out, as well as other attempts to see if it's cool to craft a dcent loss function
# if this doesn't work, I think we shuld definitely explore GANs to generate image continuations as that could be very interesting as a partial thing to see if there's anything cool there, althou this approach of crossprediction is much more plausible from a cognitive science perspective, I think, but GANs could be very interesting also!
