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
	
	#start the plot
	fig = plt.figure()
	bar(x, height, width, align=align, tick_labels=tick_labels)
	#change this depending on how it is written
	plt.title('Error of the network with data augmentation or data copying')
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
		plt.imshow(augment_test[i], cmap='gray')
		plt.title('Augmented Test image')
		plt.xticks([])
		plt.yticks([])

		ax2 = fig.add_subplot(132)
		plt.imshow(augment_preds[i], cmap='gray')
		plt.title('Augmented Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax3 = fig.add_subplot(133)
		plt.imshow(augment_errmaps[i], cmap='gray')
		plt.title('Augmented error map')
		plt.xticks([])
		plt.yticks([])

		ax4 = fig.add_subplot(231)
		plt.imshow(copy_test[i], cmap='gray')
		plt.title('Copies Test image')
		plt.xticks([])
		plt.yticks([])

		ax5 = fig.add_subplot(232)
		plt.imshow(copy_preds[i], cmap='gray')
		plt.title('Copies Image prediction')
		plt.xticks([])
		plt.yticks([])

		ax6 = fig.add_subplot(233)
		plt.imshow(copy_errmaps[i], cmap='gray')
		plt.title('Copies error map')
		plt.xticks([])
		plt.yticks([])

		fig.tight_layout()
		plt.show()


if __name__=='__main__':
	plot_training_loss('augments_training_loss.npy', 'copies_training_loss.npy')
	plot_validation_loss('augments_validation_loss.npy', 'copies_validation_loss.npy')