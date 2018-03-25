#this is where the results are calculated before plotting
#not sure it is totaly needed to be honest, but could be a useful separation of concerns

import numpy as np 
import scipy
import keras
from keras.models import load_model
from utils import *
from plotting import *


def calculate_average_error(augments_name, copies_name):
	augment_test, augment_preds, augment_his = load_array(augments_name)
	copy_test, copy_preds, copy_his = load_array(copies_name)
	augment_errmaps = get_error_maps(augment_test, augment_preds)
	copy_errmaps = get_error_maps(copy_test, copy_preds)
	augment_error = get_mean_error(augment_errmaps)
	copy_error =get_mean_error(copy_errmaps)
	return augment_error, copy_error


#I'll need something that will be able to calcualte the training curves
# from the histories
# and then obviously load the models and continue training
# so that it is possible to see the dissapearing prediction errors
#which is just the error maps
#which could be interesting!


#and some quick tests
if __name__ == '__main__':
	print "In main!"



