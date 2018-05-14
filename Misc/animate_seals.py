# okay, this is where I'm going to expeiment with the and vocal learning animation, so hopefully
# it will actualy work!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


save_name='vocal_learning_development.npy'
arr = np.load(save_name)
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


anim = animation.FuncAnimation(fig, updateFig, interval=100, blit=True, save_count=150)
#plt.show()
#save animation
anim.save('vocal_learning_animation_7.mp4',writer="ffmpeg")

# beware though, the save of the animation number and the number of the actual matrix file are opposites
# so 7 is 1 and vice versa. because I did it in a silly way!

