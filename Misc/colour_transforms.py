# just a simple model of transforms to transform one colour space into the other

from __future__ import division
import numpy as np

# all transforms assume RGB as default

def opponent_colour(mat):
	r = mat[:,:,0]
	g = mat[:,:,1]
	b = mat[:,:,2]
	#probably a better way to do this
	I = r + g + b
	RG = r - g
	BY = b - (r+g)/2
	# hopefully this concat works!
	return np.concat(np.concat(I, RG), BY)