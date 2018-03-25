import numpy as np
import matplotlib.pyplot as plt
from utils import *

#plot bar chart of the average two errors
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
