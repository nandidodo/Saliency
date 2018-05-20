# okay, this is where I'm going to expeiment with the and vocal learning animation, so hopefully
# it will actualy work!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import sys
import os


save_name='vocal_learning_development_1.npy'
print sys.argv
input_str = sys.argv[1]
if input_str is not None and type(input_str) is str:
	save_name = input_str
try:
	arr = np.load(save_name)
except ValueError:
	raise ValueError('Inputted name does not correspond to a file')
fig = plt.figure()
#ax = fig.add_subplot(111)
plt.xticks([])
plt.yticks([])
#epoch_text= fig.text(0.02, 0.90, '')

#fig.xticks([])
#fig.yticks([])

im = plt.imshow(arr[0], animated=True)
plt.title('Epoch: 0')

def updateFig(i):
	im.set_array(arr[i-1])
	#epoch_text.set_text('Epoch: ' + str(i+1))
	title = plt.title('Epoch: ' + str(i))
	print im
	print i
	#print epoch_text
	return im,

length = len(arr)
print "length: ", length
anim = animation.FuncAnimation(fig, updateFig, interval=30, blit=True, save_count=length)
#plt.show()
#save animation
"""
if input_str is not None:
	save_name = input_str.split('.')[0] + '_animation.mp4'
	print save_name
	anim.save('test_anim.mp4', writer="ffmpeg", fps=30, extra_args=['-vcodec, libx264'])
if input_str is None:
	print "in non input string branch"
	anim.save('test_anim.mp4',writer="ffmpeg", fps=30, extra_args=['-vcodec', 'libx264'])
"""
if input_str is not None:
	splits = input_str.split('.')
	if len(splits) == 2:
		save_name =input_str.split('.')[0] + '_animation.mp4'
	if len(splits) >2:
		save_name=""
		for i in xrange(len(splits)-1):
			save_name += splits[i] + '.'
		save_name += '_animation.mp4'
	anim.save(save_name,writer="ffmpeg", fps=30, extra_args=['-vcodec', 'libx264'])

# beware though, the save of the animation number and the number of the actual matrix file are opposites
# so 7 is 1 and vice versa. because I did it in a silly way!

# when rihard picks the one he wants, you might have to rerun it with the codec argument
# in there so it can be added to html5. I currently havne't got that, but I should include it at some opint!
