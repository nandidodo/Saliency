# this is where I just put really random things which are easy to find here. so we can inspect data and so forth. this isgoig to be a very freeform file. for scripting, as that's what python is meant to do.

import numpy as np
import scipy
from file_reader import *
from utils import *
import matplotlib.pyplot as plt
#import keras#
#from keras.datasets import cifar10

"""
img = read_image(50)
#print img.shape
#show_colour_splits(img)

# okay, let's try out the fft stuff
img_freq = np.fft.fft2(img)
#calculate amplitude spectum
img_amp = np.fft.fftshift(np.abs(img_freq))
#for display take the logarithm so the scale isn't so small
img_amp_display = np.log(img_amp + 0.0001)
#rescale to -1:+1 for display
img_amp_display = (((img_amp_display - np.min(img_amp_display))*2)/np.ptp(img_amp_display)) -1

print type(img_amp_display)
print img_amp_display.shape

print img
print "  "
print "  "
print img_amp_display

img_amp_display = img_amp_display *255
img_amp_display =  img_amp_display.astype('uint8')
plt.imshow(img_amp_display)
plt.show()

"""
"""
# let's test our high pass filtering
#get_amplitude_spectrum(img, show=True)
#get_magnitude_spectrum(img, show=True)
# reshape
print img.shape
# okay, let's get this as grayscale
img = img[:,:,0]
img = reshape_into_image(img)
print img.shape

#hpf = high_pass_filter(img, show=True)
#print hpf

# okay, other band tests here:
plt.imshow(img)
plt.show()
print "High Pass!"
hpf = highpass_filter(img, show=True)
print "  "
print "Low Pass"
lpf = lowpass_filter(img, show=True)
print "  "
print "Bandpass"
bpf = bandpass_filter(img, show=True)

print "both"
compare_two_images(lpf, hpf, 'Low Pass Filter', 'High Pass Filter')
"""
"""
# okay, we're going to test the filesystem stuff here
rootdir = './testSet/Stimuli'
#print_dirs_files(rootdir)
save_images_per_directory(rootdir, save=False)

"""
"""
# okay, tests on image splitting

(xtrain, ytrain), (xtest, ytest)  = cifar10.load_data()
redtrain = xtrain[:,:,:,0]
print redtrain.shape
redtrain = np.reshape(redtrain,(len(redtrain), 32,32))
print redtrain.shape
half1, half2 = split_image_dataset_into_halves(redtrain)
for i in xrange(10):
	compare_two_images(half1[i], half2[i])

"""

# tests get files saved
"""
rootdir = './BenchmarkIMAGES/SM'
cropsize = (200,200)
save_images_per_directory(rootdir, cropsize)"""

# now let's check this actuallt works, we can load them and get images, and so forth, that seems rather important to me
def load_file_test(fname):
	imgs = load(fname)
	print type(imgs)
	print imgs.shape
	plt.imshow(imgs[3])
	plt.show()

#load_file_test('BenchmarkIMAGES_images')
#load_file_test('BenchmarkIMAGES_output')

#rootdir = 'testSet/Stimuli/Action/'
#make_dir = 'testSet_Arrays_Action'
#save_images_per_directory(rootdir, save=True, crop_size=(100,100), make_dir_name=make_dir)

#load_file_test(make_dir + '/testSet_images')
#load_file_test(make_dir + '/Action_output')

#rootdir = 'BenchmarkIMAGES/'
#make_dir = 'BenchmarkDATA'
#save_images_per_directory(rootdir, save=True, crop_size=(200,200), make_dir_name=make_dir)


#rootdir = 'testSet_Arrays'
#make_dir = 'combined'
#combine_arrays_into_one(rootdir, make_dir_name=make_dir)

#load_file_test('testSet_Arrayscombined/_combined')

#imgs = load('testSet_Arrayscombined/images_combined')
#for i in xrange(20):	
#	plt.imshow(imgs[200+i])
#	plt.show()

def get_files_in_directory(dirname):
	filelist = []
	for fname in sorted(os.listdir(dirname)):
		filelist.append(fname)
		print fname
	return filelist

def get_dirs_from_rootdir(rootdir, mode='RGB', crop_size = None, save=True, save_dir=None):

	if save:
		assert save_dir is not None and type(save_dir) == str, 'save directory must exist and be a string'

	for dirs, subdirs, files in os.walk(rootdir):
		print dirs
		filelist = get_files_in_directory(dirs)
		arr = []
		for f in filelist:
			#first we check it's a file
			if '.' in f:
				#tehn we get rid of the jpg
				fname = f.split('.')[0]
				#next we check if it's output or not
				# it's output
				if crop_size is not None:
					img = imresize(imread(dirs+ '/'+f, mode=mode), crop_size)
				if crop_size is None:
					img = imread(dirs+'/'+fname, mode=mode)
				arr.append(img)

		arr = np.array(arr)
		splits = dirs.split('/')
		name='default'
		if len(splits) == 3:
			# i.e. normal file
			name= splits[-1]
		if len(splits) == 4:
			name= splits[-2] + '_' + splits[-1]
		if save:
			save_array(arr, save_dir + '/' + name)
			print "SAVING AS: " + str(save_dir + '/' + name)
				
				


# okay, the sorted is the key, without that it just breaks terribly. So we need to fix this in our functoins before it works, which should hopefully help, so let's work at that! now at least we nkow the problem. hopefully we can get some reasonable results tomorrow to show to richard, for thursday



def compare_image_and_salience(dirname, N=20, start=0):
	for i in xrange(N):
		imgs = load(dirname)
		imgs = imgs[:,:,:,0]
		shape = imgs.shape
		imgs = np.reshape(imgs, (shape[0], shape[1], shape[2]))
		print "IMGS:"
		print imgs.shape
		outputs = load(dirname+'_Output')
		shape = outputs.shape
		outputs = outputs[:,:,:,0]
		outputs = np.reshape(outputs, (shape[0], shape[1], shape[2]))
		print "OUTPUTS:"
		print outputs.shape
		compare_images((imgs[start+i], outputs[start+i]), ('image', 'salience map'))

def compare_image_and_salience_from_known_files(fname1, fname2, N=20, start=0):
	imgs = load(fname1)
	imgs = imgs[:,:,:,0]
	shape = imgs.shape
	imgs = np.reshape(imgs, (shape[0], shape[1], shape[2]))
	print "IMGS:"
	print imgs.shape
	outputs = load(fname2)
	shape = outputs.shape
	outputs = outputs[:,:,:,0]
	outputs = np.reshape(outputs, (shape[0], shape[1], shape[2]))
	print "OUTPUTS:"
	print outputs.shape
	for i in xrange(N):
		compare_images((imgs[start+i], outputs[start+i]), ('image', 'salience map'))

#compare_image_and_salience('testSet/Data/test/Action')

#compare_image_and_salience_from_known_files('testimages_combined', 'testsaliences_combined')
#
def combine_images_into_big_array(dirname, makedir = '', save=True, verbose=True):	

	if makedir != '':
		if not os.path.exists(rootdir + makedir):
			try:
				os.makedirs(rootdir + makedir)
			except OSError as e:
				if e.errno!= errno.EEXIST:
					print "error found: " + str(e)
					raise
				else:
					print "directory probably already exists despite check"
					raise
		

	filelist =  sorted(os.listdir(dirname))
	imgs = []
	outputs = []
	for f in filelist:
		arr = load(dirname + '/' + f)
		if '_' in f: #i.e. it's an output
			outputs.append(arr)
			print "OUTPUT: " + f
			print arr.shape
		if '_' not in f: # so it's an image
			imgs.append(arr)
			print "IMAGE: " + f
			print arr.shape
	#we now stack them

	imgs = np.concatenate(imgs)
	outputs = np.concatenate(outputs)
	if verbose:
		print "images shape: " + str(imgs.shape)
		print "outputs shape: " + str(outputs.shape)

	if save: 
		save_array(imgs, dirname + makedir + 'images_combined')
		save_array(outputs, dirname + makedir + 'saliences_combined')

	return imgs, outputs

#dirname = 'testSet/Data/test'
#combine_images_into_big_array(dirname)

#arr = load_array('testimages_combined_imgs_preds_errmaps')

"""
print type(arr)
print len(arr)
print arr.shape
maps = arr[2]
print maps.shape
maps = np.reshape(maps,(maps.shape[0], maps.shape[1]))
plt.imshow(maps)
plt.show()"""

#arr = load_array('error_map_test')
#maps = np.reshape(arr, (arr.shape[0], arr.shape[1], arr.shape[2]))


#arr = load_array('cifar_error_map_preliminary')
#print type(arr)
#print len(arr)
#print arr.shape
#maps = np.reshape(arr, (arr.shape[0], arr.shape[1], arr.shape[2]))


#arr = load_array('all_errmaps_imgs_preds_errmaps')
#print type(arr)
#print len(arr)
#maps = arr[2]
#print type(maps)
#print maps.shape

#for i in xrange(1):
#	plt.imshow(maps[i])
#	plt.show()
		

def sum_normalise_err_maps(errmaps):
	N = len(errmaps)
	basemap = np.zeros(errmaps[0].shape)
	print "basemap shape"
	print basemap.shape
	for i in xrange(N):
		errmap = errmaps[i]
		shape=errmap.shape
		print shape
		for j in xrange(shape[0]):
			for k in xrange(shape[1]):
				basemap[j][k] += errmap[j][k]
	
	#basemap = basemap/float(N)
	basemap = np.array(basemap)
	return basemap

#basemap = sum_normalise_err_maps(maps)
#basemap = basemap[30:70, 30:70]
#print type(basemap)
#print basemap.shape
#plt.imshow(basemap)
#plt.show()

def plot_basemap(basemap, cmap='gist_gray',save=False, fname=""):
	plt.imshow(basemap, cmap=cmap)
	plt.title("Average of all error maps")
	plt.xticks([])
	plt.yticks([])
	if not save:
		plt.show()
	if save:
		plt.savefig("results/basemap_" + fname + ".png")
#plt.imshow(errmaps[2])
#print basemap


def iterate_through_cmaps_and_save(basemap):
	cmap_string="Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r"

	cmaps = cmap_string.split(",")
	for i in xrange(len(cmaps)):
		cmap = cmaps[i].strip()
		plot_basemap(basemap,cmap=cmap, save=True, fname=cmap)



def split_by_spatial_frequency(name, save_name):
	arr = load_array(name)
	arr = arr[:,:,:,0]
	shape = arr.shape
	arr = np.reshape(arr, (shape[0], shape[1], shape[2]))
	lp,hp,bp = split_dataset_spatial_frequency(arr, save_name=save_name)
	return (lp, hp, bp)

def test_spatial_frequency_split():
	# okay, let's test it
	test = load_array('benchmark_images_spatial_frequency_split')
	print type(test)
	print len(test)
	lp, hp, bp = test
	print lp.shape
	print hp.shape
	for i in xrange(10):
		compare_two_images(lp[i], hp[i], 'lp', 'hp')






#plot_basemap(basemap)
#iterate_through_cmaps_and_save(basemap)
#save_array(basemap, "center_bias_base_error_map")

#get and save the dataset split by spatial frequency
"""
arr = load_array('testimages_combined')
# I think this only works with a single colour, so we're going to arbitrarily choose red
# which is probably bad
arr = arr[:,:,:,0]
shape=arr.shape
arr = np.reshape(arr, (shape[0], shape[1], shape[2]))
print arr.shape
# now we filter
lp,hp,bp = split_dataset_spatial_frequency(arr, save_name='benchmark_images_spatial_frequency_split')
"""






def load_show_colour_split_images(fname="testimages_combined"):
	imgs = load_array(fname)
	print imgs.shape
	for i in xrange(len(imgs)):
		img = imgs[i]
		red = img[:,:,0]
		green = img[:,:,1]
		blue = img[:,:,2]
		titles=("Original", "Red", "Green", "Blue")
		showimgs = (img, red,green,blue)
		compare_images(showimgs, titles,reshape=False)

def show_colour_split_images(fname="benchmarkIMAGES_images"):
	imgs = load_array(fname)
	for i in xrange(len(imgs)):
		img = imgs[i]
		red = img[:,:,0]
		green = img[:,:,1]
		blue = img[:,:,2]
		
		compare_two_images(img, red)

def load_and_show_colour_split_images(fname):
	img = plt.imread(fname)
	red = img[:,:,0]
	green = img[:,:,1]
	blue=img[:,:,2]
	
	"""
	#colour img
	plt.subplot(111)
	plt.imshow(img)
	plt.title('Original Colour Image')
	
	#red
	plt.subplot(212)
	plt.imshow(red)
	plt.title('Red Channel')
	
	#green
	plt.subplot(221)
	plt.imshow(green)
	plt.title('Green Channel')
	
	#blue
	plt.subplot(222)
	plt.imshow(blue)
	plt.title('Blue Channel')
	
	#show
	plt.show()
	"""
	fig = plt.figure()

	#originalcolour
	ax1 = fig.add_subplot(221)
	plt.imshow(img)
	plt.title('Original Colour Image')
	plt.xticks([])
	plt.yticks([])

	#red
	ax2 = fig.add_subplot(222)
	plt.imshow(red)
	plt.title('Red Channel')
	plt.xticks([])
	plt.yticks([])

	#green
	ax3 = fig.add_subplot(223)
	plt.imshow(green)
	plt.title('Green Channel')
	plt.xticks([])
	plt.yticks([])

	##blue
	ax4 = fig.add_subplot(224)
	plt.imshow(blue)
	plt.title('Blue Channel')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	plt.show(fig)
	

	

def show_all_error_maps_with_original_imgs(fname="testimages_combined_imgs_preds_errmaps", sal_fname="testsaliences_combined",img_nums=(180,181,182)):
	arr = load_array(fname)
	imgs, preds, errmaps = arr
	print imgs.shape
	imgshape = imgs.shape
	imgs = np.reshape(imgs, (imgshape[0], imgshape[1], imgshape[2]))
	print errmaps.shape
	#plt.imshow(imgs[180])
	#plt.show()
	sals = load_array(sal_fname)
	#print sals.shape
	#plt.imshow(sals[2])
	#plt.show()
	testsals = sals[1710:1900,:,:,:]
	print testsals.shape
	#plt.imshow(testsals[2])
	#plt.show()
	num = 151
	num2 = 152
	
	#now we plot to test if it works
	fig = plt.figure()
	
	ax1 = fig.add_subplot(231)
	plt.imshow(imgs[num], cmap='gray')
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])
	
	ax2 = fig.add_subplot(232)
	plt.imshow(gaussian_filter(errmaps[num]), cmap='gray')
	plt.title('Predicted Salience Map')
	plt.xticks([])
	plt.yticks([])
	
	ax3 = fig.add_subplot(233)
	plt.imshow(testsals[num],cmap='gray')
	plt.title('Actual Salience Map')
	plt.xticks([])
	plt.yticks([])

	ax4 = fig.add_subplot(234)
	plt.imshow(imgs[num2], cmap='gray')
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])
	
	ax5 = fig.add_subplot(235)
	plt.imshow(gaussian_filter(errmaps[num2]), cmap='gray')
	plt.title('Predicted Salience Map')
	plt.xticks([])
	plt.yticks([])
	
	ax6 = fig.add_subplot(236)
	plt.imshow(testsals[num2],cmap='gray')
	plt.title('Actual Salience Map')
	plt.xticks([])
	plt.yticks([])
	
	plt.tight_layout()
	plt.show(fig)

	




def load_imgs(fname):
	imgs = load_array(fname)
	print imgs.shape
	for i in xrange(100):
		plt.imshow(imgs[i])
		plt.show()





if __name__ =='__main__':
	#dirname = 'testSet/Stimuli/Action'
	#get_files_in_directory(dirname)

	#rootdir = 'testSet/Stimuli'
	#get_dirs_from_rootdir(rootdir, crop_size = (100, 100), save_dir = 'testSet/Data/test')

	#load_file_test('testSet/Data/test/Action')
	#load_file_test('testSet/Data/test/Action_Output')



	#show_all_error_maps_with_original_imgs()
	


	#load_show_colour_split_images()
	#show_colour_split_images()
	#load_and_show_colour_split_images("BenchmarkIMAGES/i5.jpg")




















# okay, yay, if we crop it sufficiently, we can get an okay looking centre bias heatmap, but it seems like a bit of a scam that w need to crop it sufficiently, but nevertheless, it kind of works. i don't know what other solutions i'm meant to get. so that's good. at least we are sorted now, which is great. but anyhow, how are we actually going to get this sorted? let's go and look up and catalogue a whole bunch of centre bias papers, and also figure out just what we're doing here, let's do that until mycah finishes work. from there, the next step is to have lunch and finish this stuff off. hopefully we can present richard, perhaps even with a draft paper next thursday on the centre bias, of a thousand words or so, plus our stuff and an exlpanation of the results. thiscould be very reasonable indeed. and we'll take the best things and compare them, although we have no actual accuracy/ROC calculation metrics here, which is unfortuante, but that's still quitemiportant, so we should seriously look at that and see what's up, and we also need to perform hyperparameter optimisation and stuff on our models, to get it significantly better, and like use the ADAM optimiser and so forth, so we shuold look at that and also at other people's approaches. so let's get on with that then right now!

		


#plt.imshow(img_amp_display)
#plt.show()
"""
error_maps = load('error_map_test')
error_map = error_maps[1]
print error_map.shape
err = np.reshape(error_map, [28,28])
print err.shape
plt.imshow(err)
plt.show()"""

"""
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
print xtrain.shape

red = xtrain[:,:,:,0]
print red.shape
redimg = red[5]
print redimg.shape


plt.imshow(xtrain[5])
plt.show()

plt.imshow(redimg)
plt.show()
"""






"""
a = [[1,5,3],[4,5,7],[7,5,2]]
a = np.array(a)
print a
print a.shape
print np.argmax(a)
print np.argmax(a, axis=0)
print np.argmax(a, axis=1)"""




