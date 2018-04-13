import numpy as np
import matplotlib.pyplot as plt
from utils import *

#plot bar chart of the average two errors

def plot_training_loss(aug_losses,copy_losses):
	aug_losses = np.load(aug_losses)
	copy_losses = np.load(copy_losses)

	print aug_losses.shape
	print copy_losses.shape

	fig = plt.figure()
	assert len(aug_losses) == len(copy_losses), 'Two losses must have gone on for same number of eopchs'
	N = np.linspace(0, len(aug_losses),num=10)
	print len(N)
	plt.plot(N,aug_losses,label='Training loss with augmented data')
	plt.plot(N,copy_losses, label='Training loss with copied data')
	plt.xlabel('Epochs')
	plt.ylabel('Training loss')
	plt.title('Training loss over time with augmented data and copied data')
	plt.legend()
	fig.tight_layout()
	plt.show()
	return fig


def plot_validation_loss(aug_losses,copy_losses):
	aug_losses = np.load(aug_losses)
	copy_losses = np.load(copy_losses)

	fig = plt.figure()
	assert len(aug_losses) == len(copy_losses), 'Two losses must have gone on for same number of eopchs'
	N = np.linspace(0, len(aug_losses), num=10)
	plt.plot(N,aug_losses,label='Validation loss with augmented data')
	plt.plot(N,copy_losses, label='Validation loss with copied data')
	plt.xlabel('Epochs')
	plt.ylabel('Validation loss')
	plt.title('Validation loss over time with augmented data and copied data')
	plt.legend()
	fig.tight_layout()
	plt.show()
	return fig


def average_error_bar_chart(save_name):
	augment_error, copy_error = load_array(save_name)
	x = [0,1]
	height = [augment_error, copy_error]
	width=0.7
	align='center'
	tick_labels = ('Mean Error with Augmentation', 'Mean Error with Copying')
	linewidth = 0.1
	#start the plot
	fig = plt.figure()
	plt.bar(x, height, width, align=align, tick_label=tick_labels, linewidth=linewidth)
	#change this depending on how it is written
	plt.title('Error of the network with data augmentation or data copying')
	fig.tight_layout()
	plt.show()
	return fig

#okay, I need to plot all the fixations on oen graph to see what's up, hopefully it will work
#plot 12 for the fading and 12 for the non fading?

def plot_graph_loop(imgs, num_rows, imgs_per_row, item_label, cmap='gray'):
	if len(imgs)< num_rows*imgs_per_row:
		raise ValueError('Not enough images supplied for graphing')
	fig = plt.figure()
	for i in xrange(num_rows):
		for j in range(imgs_per_row):

			#hope this is right but probably not as I have brain damage
			epoch = (i*imgs_per_row)+(j+1)
			print epoch
			img = imgs[epoch][2] # since you're getting the errmaps
			print img.shape
			ax = fig.add_subplot(num_rows, imgs_per_row, epoch)
			plt.imshow(imgs[epoch])
			plt.title(item_label + ' Epoch: ' + str(epoch))
			plt.xticks([])
			plt.yticks([])

	#I just can't think properly, dagnabbit, and I feel sick. ugh, I'm really struggling with this
	# and I'm just fighting against carbon monoxide poisoning /brian damage... argh!
	plt.suptitle('Prediction errors of ' + item_label +' Condition being fixated over time', fontsize=14)
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.6)
	plt.show()
	return fig

	#who knew this was so horrendously difficlt just to get the requisite plots. 
	# I also need to do the data analysis of the drift results
	# and then write the whole discussion and conclusion, why isthi so difficult
	 #


def plot_12_fixation_errmaps_one_graph_both(fixation_results_augments, fixation_results_copy,N=None, save_name =None):
	#not totally sure what the point of N is, but hey, I think it's so we can see them all, let's try it out

	if N is None:
		N = len(fixation_results_augments)

	#just test this is the start
	if type(fixation_results_augments) is str:
		fixation_results_augments = load(fixation_results_augments)
	if type(fixation_results_copy) is str:
		fixation_results_copy = load(fixation_results_copy)

	#this is the sort of careless stupidity which could be caused by brain damage argh!
	if len(fixation_results_copy)!=len(fixation_results_augments):
		raise ValueError('Copies and augments should have same length: Augments Length:' + str(len(fixation_results_augments)) + ' Copies length: ' +str(len(fixation_results_copy)))


		#this is the sort of other stupid mistakes caused by my weird feelin headache and brain damage... dagnabbit
		# I would never have made that mistake before. I don't know what is wrong with me
	print type(fixation_results_copy)
	print len(fixation_results_copy)
	print type(fixation_results_augments)
	print len(fixation_results_augments)
	bib = fixation_results_augments[0]
	tib = fixation_results_copy[2]
	print type(bib)
	print len(bib)
	print type(tib)
	print len(tib)
	b = bib[0]
	print type(b)
	print len(b)

	run_num = 12 # assume 12
	aug_fig = plot_graph_loop(fixation_results_augments, 4,3,'Augments')
	copy_fig = plot_graph_loop(fixation_results_copy, 4,3,'Copies')






def plot_fixation_errmaps(fixation_results_augments, fixation_results_copy, N = None):
	if N is None:
		N = len(fixation_results_augments)
		assert N == len(fixation_results_copy), 'Fixation results must be same length'
	for i in xrange(N):
		aug_tests, aug_preds, aug_errmaps = fixation_results_augments[i]
		copy_tests, copy_preds, copy_errmaps = fixation_results_copy[i]
		print np.sum(aug_errmaps)
		print np.sum(copy_errmaps)
		# sum is less, s that's good
		



		#just get the first one, because why not here
		aug_test = aug_tests[0]
		aug_pred = aug_preds[0]
		aug_errmap = aug_errmaps[0]

		copy_test = copy_tests[0]
		copy_pred = copy_preds[0]
		copy_errmap = copy_errmaps[0]

		concat = np.concatenate((aug_errmap,copy_errmap))
		print concat.shape
		sh = concat.shape
		concat = np.reshape(concat, (sh[0],sh[1]))

		#print aug_errmap[10][10]
		#print copy_errmap[10][10]
		# okay, these are the same. all the results are the same... but why?


		#reshapes!
		sh = aug_test.shape
		print sh
		#assume all others are of same shape
		aug_test = np.reshape(aug_test, (sh[0], sh[1]))
		aug_pred = np.reshape(aug_pred, (sh[0], sh[1]))
		aug_errmap = np.reshape(aug_errmap, (sh[0], sh[1]))

		copy_test = np.reshape(copy_test, (sh[0], sh[1]))
		copy_pred = np.reshape(copy_pred, (sh[0], sh[1]))
		copy_errmap = np.reshape(copy_errmap, (sh[0], sh[1]))

		fig = plt.figure()

		ax1 = fig.add_subplot(131)
		plt.imshow(aug_test, cmap='gray')
		plt.title('Augmented Test image')
		plt.xticks([])
		plt.yticks([])

		ax2 = fig.add_subplot(132)
		plt.imshow(aug_pred, cmap='gray')
		plt.title('Augmented Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax3 = fig.add_subplot(133)
		plt.imshow(aug_errmap, cmap='gray')
		plt.title('Augmented error map')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(231)
		plt.imshow(copy_test, cmap='gray')
		plt.title('Copies Test image')
		plt.xticks([])
		plt.yticks([])

		ax5 = fig.add_subplot(232)
		plt.imshow(copy_pred, cmap='gray')
		plt.title('Copies Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(233)
		plt.imshow(copy_errmap, cmap='gray')
		plt.title('Copies error map')
		plt.xticks([])
		plt.yticks([])

		fig.tight_layout()
		plt.show()

		fig = plt.figure()
		plt.imshow(concat, cmap='gray')
		plt.title('Concatenated')
		plt.show()

		# the trouble with this is that it decreases too fast
		# so I'm going to doa single epoch instead this time in the hope of a better success there
		# to see it actually dissapearing out from epochs, so that's the hope!
		# hopefully it will work that

def plot_generative_invariance_bar_chart(base_fname):
	aug_errs = np.load(base_fname+'_aug.npy')
	copy_errs = np.load(base_fname+'_copy.npy')
	pixels = np.load(base_fname+'_pixels.npy')

	w =0.4
	align='center'

	#begin plot
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.bar(pixels-w, aug_errs, width=w, align=align, label='Augmented error')
	ax.bar(pixels+w, copy_errs, width=w, align=align, label='Copy errors')
	plt.title('Error of copy and augmented against number of pixels translated')
	plt.xlabel('Pixels translated')
	plt.ylabel('Mean error')
	plt.legend()
	fig.tight_layout()
	plt.show()
	return fig

def plot_discriminative_invariance_bar_chart(base_fname):
	augs = np.load(base_fname+'_aug_accuracies.npy')
	copies = np.load(base_fname+'_copy_accuracies.npy')
	pixels = np.load(base_fname+'_pixels.npy')

	w = 0.4
	align='center'
	#begin plot
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.bar(pixels-w, augs, width=w, align=align, label='Augmented accuracy')
	ax.bar(pixels+w, copies, width=w, align=align, label='Copy accracy')
	plt.title('Accuracies of copy and augmented against number of pixels translated')
	plt.xlabel('Pixels translated')
	plt.ylabel('Mean percentage accuracy')
	plt.legend()
	fig.tight_layout()
	plt.show()
	return fig

def plot_errmaps(augments_name, copies_name, N =20):
	augment_test = np.load(augments_name + '_test.npy')
	augment_preds = np.load(augments_name+'_preds.npy')

	copy_test = np.load(copies_name + '_test.npy')
	copy_preds = np.load(copies_name + '_preds.npy')

	augment_errmaps = get_error_maps(augment_test, augment_preds)
	copy_errmaps = get_error_maps(copy_test, copy_preds)
	augment_error = get_mean_error(augment_errmaps)
	copy_error =get_mean_error(copy_errmaps)

	#reshape so work as images
	sh = augment_test.shape # all shapes should be the same here!
	augment_test = np.reshape(augment_test, (sh[0], sh[1],sh[2]))
	augment_preds= np.reshape(augment_preds, (sh[0], sh[1],sh[2]))
	augment_errmaps = np.reshape(augment_errmaps, (sh[0], sh[1],sh[2]))

	copy_test = np.reshape(copy_test, (sh[0], sh[1],sh[2]))
	copy_preds = np.reshape(copy_preds, (sh[0], sh[1],sh[2]))
	copy_errmaps = np.reshape(copy_errmaps, (sh[0], sh[1],sh[2]))

	for i in xrange(N):
		#begin the plot
		fig = plt.figure()

		ax1 = fig.add_subplot(131)
		plt.imshow(augment_test[i], cmap='gray', vmin=0, vmax=255)
		plt.title('Augmented Test image')
		plt.xticks([])
		plt.yticks([])

		ax2 = fig.add_subplot(132)
		plt.imshow(augment_preds[i], cmap='gray', vmin=0, vmax=255)
		plt.title('Augmented Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax3 = fig.add_subplot(133)
		plt.imshow(augment_errmaps[i], cmap='gray', vmin=0, vmax=255)
		plt.title('Augmented error map')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(231)
		plt.imshow(copy_test[i], cmap='gray', vmin=0, vmax=255)
		plt.title('Copies Test image')
		plt.xticks([])
		plt.yticks([])

		ax5 = fig.add_subplot(232)
		plt.imshow(copy_preds[i], cmap='gray', vmin=0, vmax=255)
		plt.title('Copies Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(233)
		plt.imshow(copy_errmaps[i], cmap='gray', vmin=0, vmax=255)
		plt.title('Copies error map')
		plt.xticks([])
		plt.yticks([])

		fig.tight_layout()
		plt.show()


if __name__=='__main__':
	#plot_training_loss('augments_training_loss.npy', 'copies_training_loss.npy')
	#plot_validation_loss('augments_validation_loss.npy', 'copies_validation_loss.npy')
	#average_error_bar_chart('errors_1')
	#plot_generative_invariance_bar_chart('results/generative_invariance')
	#plot_discriminative_invariance_bar_chart('results/discriminative_invariance_2')
	plot_12_fixation_errmaps_one_graph_both('results/from_scratch_fixation_augments', 'results/from_scratch_fixation_copy')
	# if copy performs better in the copy position, could argue as to why suppressed
	# but they may not even be suppressed and whether that is the case is doubtful
	# but it would be a good explanation for why it occurs, why it would be useful
	# or wahtever, but they don't sem to be random but instead modulated by task demands.

	# so this did not work precisely as hoped, the resulst are strange, error decreases(!)
	# further from the original image. not sure what is going there
	# I think I'm going to have to train the discriminative network instead
	# and hope that that works better. I should look into doing that now
	# as it won't be at all difficult most likely, so let's look into doing that
	# while I read the papers necessary or whatever!