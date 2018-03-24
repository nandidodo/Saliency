
#some simple utils for things

def get_error_map(original_img, pred):
	assert original_img.shape ==pred.shape, 'Original image and prediction must have same shape'
	#for the moment just do a subtraction
	return original_img - pred

def get_total_error(error_map):
	return sum(error_map)

