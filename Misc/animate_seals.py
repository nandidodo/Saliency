# okay, this is where I'm going to expeiment with the and vocal learning animation, so hopefully
# it will actualy work!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


save_name='vocal_learning_development.npy'
arr = np.load(save_name)
fig = plt.figure
fig.xticks([])
fig.yticks([])

im = plt.imshow(arr[0], animated=True)

def updateFig(i):
	im.set_array(arr[i+1])
	return im,

anim = animation.FuncAnimation(fig, updateFig, interval=50, blit=True)
plt.show()

