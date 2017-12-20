#this is just various scripts for loading different datasets, so it makes perfect sense here
# so let's do this

#for the MIT
import sys
#add our prev folder to the path
sys.path.append('/home/beren/work/phd/saliency')
from file_reader import *
if len(sys.argv) ==2:
	dataset_name = sys.argv[1]



def save_MIT_dataset():
	basepath = "/home/beren/work/phd/saliency/datasets/MIT/"
	fixmaps = basepath+"/ALLFIXATIONMAPS"
	stimuli = basepath+"/ALLSTIMULI"
	#save fixations
	save_images_per_directory(fixmaps, save_dir=basepath + "_fixation_maps")
	#save images
	save_images_per_directory(stimuli, save_dir=basepath + "_images")



def check_datasets_to_run():
	if dataset_name=='MIT':
		save_MIT_dataset()



if __name__ == '__main__':
	save_MIT_dataset()
