

import numpy as np
import scipy

def translate(img, px, mode='constant'):
	assert type(px)==float or type(px)==int or len(px)==len(img.shape),'Pixels must either be a value or a tuple containing the number of pixels to move for each axis, and therefore musth ave the same dimension as the input image'
	return scipy.ndimage.interpolation.shift(img, px, mode=mode)


def augment_with_translations(img, num_augments=10,max_px_translate=4):
	augments = []
	#add the initial iamge
	augments.append(img)
	for i in range(num_augments):
		#get pixel values
		x_shift = int(np.random.uniform(low=(-1*max_px_translate), high=max_px_translate))
		y_shift = int(np.random.uniform(low=(-1*max_px_translate), high=max_px_translate))
		augment = translate(img, (x_shift, y_shift))
		augments.append(augment)

	#turn into numpy array and return
	augments = np.array(augments)
	return augments




