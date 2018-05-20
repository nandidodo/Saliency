from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy
import scipy.stats
import os

def euclidean_distance(center, point):
	if len(center)!=len(point):
		raise ValueError('Point and center must have same dimensionality')
	total = 0
	for i in xrange(len(center)):
		total += (center[i] - point[i])**2
	return np.sqrt(total)


def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def create_random_colour_matrix(height, width):
	mat = np.zeros((height, width,3))
	for i in xrange(height):
		for j in xrange(width):
			mat[i][j][0] = (np.random.uniform(low=0, high=1) * 255.)
			mat[i][j][1] = np.random.uniform(low=0, high=1) * 255.
			mat[i][j][2] = np.random.uniform(low=0, high=1) * 255.
	return mat

def average_point(mat,center,px_radius, image_height, image_width):
	x,y = center
	green_total = 0
	red_total= 0
	blue_total = 0
	number = 0
	for i in xrange(px_radius*2):
		for j in xrange(px_radius*2):
			#print "going round loop: " + str(i) + " " + str(j) +" " + str(x) + " " + str(y)
			#print x-px_radius+i 
			#print y-px_radius +j
			xpoint = x - px_radius + i
			ypoint = y - px_radius + j
			#print euclidean_distance(center, (i,j))
			#check it falls within bounds, then check euclidena distance
			if xpoint >=0 and xpoint < image_height:
				if ypoint >=0 and ypoint + j <image_width:
					if euclidean_distance(center, (xpoint, ypoint)) <=px_radius:
						#print "adding to average"
						green_total+= mat[xpoint][ypoint][0]
						#print "green added: " + str(mat[xpoint][ypoint][0])
						#print "green total: " + str(green_total)
						red_total+= mat[xpoint][ypoint][1]
						#print "red added: " + str(mat[xpoint][ypoint][1])
						blue_total+=mat[xpoint][ypoint][2]
						number+=1

	#print "number: ", number
	#print "green: " + str(green_total/number)
	#print "green total: " + str(green_total)
	return (green_total/number, red_total/number, blue_total/number)


def update_point(mat, center, px_radius, image_height, image_width, learning_rate=0.01):
	x,y = center
	currents = mat[x][y]
	new_points = average_point(mat, center, px_radius, image_height, image_width)
	diff = currents-new_points
	currents = currents + (learning_rate*diff)
	return currents

def create_random_mask(shape, multiplier):
	if len(shape)!=3:
		raise ValueError('Shape must be three dimensional for colour image')
	height,width,channels = shape
	return multiplier * np.random.randn(height, width, channels)

def matrix_average_step(mat, average_radius, copy=True, random_multiplier=None):
	if len(mat.shape)!=3 or mat.shape[2]!=3:
		raise ValueError('Matrix must be 2d colour image with 3 channels in format h,w,ch')

	height,width, channels = mat.shape
	if not copy:
		new_mat = mat
	if copy:
		new_mat = np.copy(mat)
	#copy so don't mutate on each run through - I can change this behaviour later if I want
	for i in xrange(height):
		for j in xrange(width):
			new_mat[i][j] = average_point(mat, (i,j), average_radius, height,width)
	if random_multiplier is not None:
		rand = create_random_mask((height,width,channels), random_multiplier)
		#print rand
		#print new_mat[40][20]
		#print rand[40][20]
		new_mat = new_mat + rand
		#print new_mat[40][20]
		#print np.amax(rand)
		# I don't udnerstand why adding a small random peturbation almost completely
		# foils the random field at all. I really don't understand that and it confuses me
		# like there seems to be no reason for it, and it confuses me so much!
		# the randomisation seems much much much greater than I would think reasonable
		# so Ihoenstly don't know!
		#
	
	return new_mat

def matrix_update_step(mat, radius, copy=True, learning_rate=0.1):

	if len(mat.shape)!=3 or mat.shape[2]!=3:
		raise ValueError('Matrix must be 2d colour image with 3 channels in format h,w,ch')

	height,width, channels = mat.shape
	if not copy:
		new_mat = mat
	if copy:
		new_mat = np.copy(mat)
	#copy so don't mutate on each run through - I can change this behaviour later if I want
	for i in xrange(height):
		for j in xrange(width):
			new_mat[i][j] = update_point(mat, (i,j), radius, height,width,learning_rate=learning_rate)

	return new_mat

def plot_image_changes(N=150, radius=5, plot_after=5, multiplier=0, save_after=1, save_name=None):
	orig_mat = create_random_colour_matrix(50,50)
	#plt.imshow(orig_mat)
	#plt.show()
	save_list = []
	print "updating with radius: " + str(radius)
	if save_name is not None:
		#reshape the orig mat to form the base of the ultimate npmpy array
		# or I could do it as a list and reshape - that's probably the best
		save_list.append(orig_mat)

	for i in xrange(N):
		orig_mat = matrix_average_step(orig_mat, radius,random_multiplier=multiplier)
		print "plot: ", i
		if save_name is not None and i % save_after ==0:
			save_list.append(orig_mat)
		#if i % plot_after ==0:
			#plt.imshow(orig_mat)
			#plt.xticks([])
			#plt.yticks([])
			#plt.show()

	if save_name is not None:
		save_list = np.array(save_list)
		#print save_list.shape
		np.save(save_name, save_list)
	return orig_mat


def get_gradient_matrix(N=20, radius=5, plot=True, save_name=None):
	orig_mat = create_random_colour_matrix(50,50)
	for i in xrange(N):
		orig_mat = matrix_average_step(orig_mat, radius)

	if plot:
		plt.imshow(orig_mat)
		plt.show()

	if save_name:
		np.save(save_name, orig_mat)

	return orig_mat

def select_random_point(mat):
	h,w,ch = mat.shape
	selected = False
	while selected != True:
		height = int(h * np.random.uniform(low=0, high=1))
		width = int(w*np.random.uniform(low=0, high=1))
		if check_proposed_points((height, width), h,w):
			selected=True
	return height,width

def select_target(mat):
	#basicaly selects at random a point
	height, width = select_random_point(mat)
	return mat[height][width]

def select_random_edge_point(mat):
	h,w,ch = mat.shape
	#four edges so essentially pick one randomly, then do up to shape
	edge = 4*np.random.uniform(low=0, high=1)
	if edge<=1:
		rand = int(h*np.random.uniform(low=0, high=1))
		return (rand,0)
	if edge>1 and edge<=2:
		rand = int(h*np.random.uniform(low=0, high=1))
		return (rand, (w-1))
	if edge>2 and edge<=3:
		rand = int(w*np.random.uniform(low=0, high=1))
		return (0, rand)
	if edge>3 and edge<=4:
		rand = int(w*np.random.uniform(low=0, high=1))
		return ((h-1), rand)




def check_proposed_points(points, height,width):
	h,w = points
	if h>=0 and h<height:
		if w>= 0 and w<width:
			#print "in check: " + str(height) + " " + str(width) + " " + str(points)
			return True
	return False

def random_walk_step(mat, initial_point, step_size):
	sh,sw = initial_point
	h,w,ch = mat.shape
	valid=False
	#init coords to always be wrong
	coords = (-5,-5)
	while valid is False:
		direction = int(8*np.random.uniform())
		if direction == 0:
			coords =  sh+step_size, sw-step_size
		if direction==1:
			coords =  sh+step_size, sw
		if direction==2:
			coords =  sh+step_size, sw+step_size
		if direction==4:
			coords =  sh, sw+step_size
		if direction==5:
			coords =  sh-step_size, sw+step_size
		if direction==6:
			coords =  sh-step_size, sw
		if direction==7:
			coords =  sh-step_size, sw-step_size
		if direction==8:
			coords =  sh, sw-step_size

		if check_proposed_points(coords, h,w) is True:
			valid=True

	return coords


def absolute_diff(p1,p2):
	if len(p1)!=len(p2):
		raise ValueError('Points to be compared must be of same dimension')
	total = 0
	for i in xrange(len(p1)):
		total += np.abs(p1[i] - p2[i])
	return total/len(p1)

def immediate_gradient_step(ideal, center, mat):
	
	best_diff = 99999 # a large number!
	# calculate differences by euclidean differences here
	ch,cw = center
	best_coords = None
	h,w,channels = mat.shape


	for i in xrange(2):
		for j in xrange(2):
			xpoint = ch+i -1
			ypoint = cw + j -1
			if xpoint >=0 and xpoint<=w:
				if ypoint>=0 and xpoint<=h:
					val = mat[xpoint][ypoint]
					diff = euclidean_distance(ideal, val)
					if diff<best_diff:
						best_diff=diff
						best_coords = (xpoint, ypoint)
			#if gradiet is the same do a random walk with a large step size
	if best_coords == center:
		coords = random_walk_step(mat, center,1)
		diff = euclidean_distance(ideal, mat[coords])
		return coords, diff

	return best_coords, best_diff


def plot_path(coords, height, width,plot=True, base=None):
	if base is None:
		base = np.zeros((height,width))
	for i in xrange(len(coords)):
		x,y = coords[i]
		base[x][y] = 255.
	if plot:
		plt.imshow(base)
		plt.xticks([])
		plt.yticks([])
		plt.show()
	return base


#yeah, thi sfunction needs a lot of work!
def plot_example_gradient_and_random(random_base, gradient_base):
	fig = plt.figure()
	plt.title('Example Random and Gradient following paths through the colony')
	ax1 = fig.add_subplot(121)
	plt.imshow(random_base)
	plt.xticks([])
	plt.yticks([])
	ax2 = fig.add_subplot(122)
	plt.imshow(gradient_base)
	plt.xticks([])
	plt.yticks([])

	fig.tight_layout()
	plt.show()

def position_base_on_path(path, base):
	new_base = np.copy(base)
	#if base.shape != path.shape:
	#	raise ValueError('Base and path are not the same shape')
	h,w = path.shape
	for i in xrange(h):
		for j in xrange(w):
			if path[i][j] >1:
				new_base[i][j] = 255.
	return new_base


def plot_example_random_levy_gradient(random_base, levy_base, gradient_base, base=None):
	fig = plt.figure()

	if base is not None:
		random_base = position_base_on_path(random_base, base)
		levy_base = position_base_on_path(levy_base, base)
		gradient_base = position_base_on_path(gradient_base, base)

	ax1 = plt.subplot(131)
	plt.imshow(random_base)
	ax1.set_title('Random walk path')
	plt.xticks([])
	plt.yticks([])

	ax2 = plt.subplot(132)
	plt.imshow(levy_base)
	ax2.set_title('Levy Flight path')
	plt.xticks([])
	plt.yticks([])

	ax3 = plt.subplot(133)
	plt.imshow(gradient_base)
	ax3.set_title('Gradient path')
	plt.xticks([])
	plt.yticks([])

	#make there be no space between subplots
	plt.subplots_adjust(wspace=0, hspace=0)
	fig.tight_layout()
	plt.show()





def gradient_search_till_atop(mat, less_diff=0.11, save_name=None, plot=False,return_base=False):

	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	#initialise random point
	ideal = mat[select_random_point(mat)]
	#initialise position
	#position = select_random_point(mat)
	position = select_random_edge_point(mat)
	#initialise to high value
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	while diff > less_diff and tries <= max_tries:

		new_coords, diff = immediate_gradient_step(ideal, position,mat)
		diffs.append(diff)
		coords.append(new_coords)
		position = new_coords
		tries +=1
		#print "num tries: " + str(tries)
		#print "diff: " + str(diff)
		#print "coords: " + str(new_coords)

	if save_name is not None:
		save((diffs, coords), save_name)

	base = None
	if plot:
		base = plot_path(coords, h,w)

	if return_base is True:
		return diffs, coords ,base
	return diffs, coords

def random_walk_till_atop(mat, less_diff=0.1, step_size=1,save_name=None, plot=False, return_base=False):
	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	#initialise random point
	ideal = mat[select_random_point(mat)]
	#initialise position
	#position = select_random_point(mat)
	position = select_random_edge_point(mat)
	#initialise to high value
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	while diff > less_diff and tries <= max_tries:

		new_coords = random_walk_step(mat,position, step_size=step_size)
		#print "coords: " + str(new_coords)
		diff = euclidean_distance(mat[new_coords], ideal)
		diffs.append(diff)
		coords.append(new_coords)
		position = new_coords
		tries +=1
		#print "num tries: " + str(tries)
		#print "diff: " + str(diff)

	if save_name is not None:
		save((diffs, coords), save_name)

	base = None
	if plot:
		base = plot_path(coords, h,w)
	print len(coords)
	print len(diffs)
	if return_base is True:
		return diffs, coords ,base

	return diffs, coords

def power_law_sample(alpha):
	samps = np.random.uniform(low=0, high=1, size=1)
	return np.power((1-samps), (-1/alpha-1))

def step_in_direction(mat, position, current_direction,step_size=1):
	#assumes step size is one really
	dirh, dirw = current_direction
	curh, curw = position
	h,w,ch = mat.shape

	coords = (curh+dirh, curw+dirw)
	if check_proposed_points(coords, h,w) is True:
		return coords
	else:
		coords = random_walk_step(mat, position, step_size=step_size)
		return coords



def levy_flight_till_atop(mat, less_diff=0.1, alpha=50, save_name=None, plot=False, return_base=False):
	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	#initialise random point
	ideal = mat[select_random_point(mat)]
	#initialise position
	#position = select_random_point(mat)
	position = select_random_edge_point(mat)
	#initialise to high value
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	curr_num_tries = 0
	current_direction = None

	while diff > less_diff and tries <= max_tries:
		#print "curr num tries: " , curr_num_tries
		if curr_num_tries<=0 or current_direction is None:
			step_size = int(power_law_sample(alpha))
			#print "STEP_SIZE:", step_size
			#break
			curr_num_tries = step_size -1
			new_coords = random_walk_step(mat, position,step_size=1)
			diff = euclidean_distance(mat[new_coords], ideal)
			diffs.append(diff)
			coords.append(new_coords)
			dirx = position[0] - new_coords[0]
			diry = position[1] - new_coords[1]
			position = new_coords
			tries +=1
			current_direction = (dirx, diry)
			#print "num tries: " + str(tries)
			#print "diff: " + str(diff)
		if curr_num_tries>0:
			#print position, current_direction
			new_coords = step_in_direction(mat, position,current_direction,step_size=1)
			diff = euclidean_distance(mat[new_coords], ideal)
			diffs.append(diff)
			coords.append(new_coords)
			position = new_coords
			curr_num_tries -=1
			tries+=1
			#print "num tries: " + str(tries)
			#print "diff: " + str(diff)

	if save_name is not None:
		save((diffs, coords), save_name)

	base = None
	if plot:
		base = plot_path(coords, h,w)
	print len(coords)
	print len(diffs)
	if return_base is True:
		return diffs, coords ,base
	return diffs, coords



def run_trial(N, step_fn, mat, less_diff=0.1, results_save=None, info=True):
	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Input matrix must be three dimensinoal and with three colour channels')
	
	h,w,ch = mat.shape
	all_coords = []
	all_diffs = []
	successes = []
	num_failures = 0
	num_successes = 0

	nums_till_success = []

	for i in xrange(N):
		diffs, coords = step_fn(mat, less_diff=less_diff,plot=False)
		n = len(diffs)
		assert n==len(coords), 'Something wrong here: differences and coordinates different lengths'
		if n<1000:
			num_successes+=1
		if n >=1000:
			num_failures+=1
		all_coords.append(coords)
		all_diffs.append(diffs)
		#if n<1000:
		assert len(coords) == len(diffs), 'Number of coordinates and differences is different'
		nums_till_success.append(len(coords))


	if results_save is not None:
		save_array(results_save+'_coords', all_coords)
		save_array(results_save+'_diffs',all_diffs)

	print "Number of failures = ", num_failures
	print "Number of success = " , num_successes

	nums_till_success = np.array(nums_till_success)

	return nums_till_success


def plot_random_vs_gradient(randoms, gradients):
	rand_mu = np.mean(randoms)
	rand_var = np.var(randoms)
	gradient_mu = np.mean(gradients)
	gradient_var = np.var(gradients)
	# the variances are for possible error bars!
	fig  = plt.figure()

	plt.bar(rand_mu, label='Mean number of steps using a random walk')
	plt.bar(gradient_mu, label='Mean number of steps using a gradient search')
	fig.xlabel('Random walk or gradient search')
	fig.ylabel('Mean number of steps to reach target')
	plt.legend()
	fig.tight_layout()
	plt.show()

def plot_random_gradient_levys(randoms, gradients, levys):
	rand_mu = np.mean(randoms)
	gradient_mu = np.mean(gradients)
	levy_mu = np.mean(levys)
	rand_stderr = np.sqrt(np.var(randoms))/np.sqrt(len(randoms))
	gradient_stderr = np.sqrt(np.var(gradients))/np.sqrt(len(gradients))
	levy_stderr = np.sqrt(np.var(levys))/np.sqrt(len(levys))

	print "supposed standard deviations"
	print rand_stderr
	print gradient_stderr
	print levy_stderr

	print "confidence intervals"
	print scipy.stats.norm.interval(0.95, loc=rand_mu, scale=np.sqrt(np.var(randoms))/np.sqrt(len(randoms)))
	print scipy.stats.norm.interval(0.95, loc=gradient_mu, scale=np.sqrt(np.var(gradients))/np.sqrt(len(randoms)))
	print scipy.stats.norm.interval(0.95, loc=levy_mu, scale=np.sqrt(np.var(levys))/np.sqrt(len(randoms)))

	print len(randoms)
	#labels = ['Random', 'Levy','Gradient']
	#try with matplotlibs latex functionality
	labels = [r'$Random$', r'$L\grave{e}vy$',r'$Gradient$']
	errors = [rand_stderr, gradient_stderr, levy_stderr]

#plt.title(r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{i}\bar{A}\tilde{n}\vec{q}$')
	pos = [1,2,3]
	res = [rand_mu, levy_mu, gradient_mu]
	fig, ax = plt.subplots()
	ax.bar(pos, res, width=0.6, yerr=errors, align='center', alpha=0.8, ecolor='black', capsize=10)
	ax.set_xlabel('Search Strategy')
	ax.set_ylabel('Mean number of steps to reach target')
	ax.yaxis.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()


def t_test(randoms, gradients):
	t,prob = scipy.stats.ttest_ind(randoms, gradients, equal_var=False)
	return t,prob

# okay, let's try to do something actually productive by using the animations to figure 
# out what's happening here!
# basicaly, all I should theoretically have to do is to is to plot it reasonably here
# I'm not sure if it's possible, but idealy it would be!


if __name__ == '__main__':
	#plot_image_changes()
	#mat = get_gradient_matrix(N=30, radius=5,save_name='gradient_matrix')
	#mat = np.load('gradient_matrix.npy')
	##np.save('matrix_1',mat)
	#mat = np.load('matrix_1.npy')
	#plt.imshow(mat)
	#plt.show()
	# the current one is not a bad one!
	# let's try this again!
	#diffs, coords, base = levy_flight_till_atop(mat, save_name='levy_flight_search',plot=True,return_base=True)
	#np.save('levy_flight_base', base)
	#diffs, coords, base = gradient_search_till_atop(mat,save_name='gradient_search_path', plot=True,return_base=True)
	#np.save('gradient_base',base)
	#diffs, coords, base = random_walk_till_atop(mat, save_name='random_walk_search', plot=True,return_base=True)
	#np.save('random_walk_base', base)
	#random_nums = run_trial(10000, random_walk_till_atop, mat, less_diff=0.1)
	#levy_nums = run_trial(10000, levy_flight_till_atop,mat, less_diff=0.1)
	#gradient_nums = run_trial(10000, gradient_search_till_atop, mat, less_diff=0.1)
	#np.save('trial_random', random_nums)
	#np.save('trial_gradient', gradient_nums)
	#np.save('trial_levy', levy_nums)
	#print len(random_nums)
	#print len(gradient_nums)
	#print "mean random: ", np.mean(random_nums)
	#print "gradient nums: " , np.mean(gradient_nums)
	#print "random variance", np.var(random_nums)
	#print "gradient variance: ", np.var(gradient_nums)
	
	"""
	rands = np.load('trial_random.npy')
	gradients = np.load('trial_gradient.npy')
	levys = np.load('trial_levy.npy')
	print "means"
	print np.mean(rands)
	print np.mean(gradients)
	print np.mean(levys)
	print "variances"
	print np.var(rands)
	print np.var(gradients)
	print np.var(levys)

	print "t-test rands gradients"
	t,prob = t_test(rands, gradients)
	print t 
	print prob
	print "t-test rands levys"
	t,prob = t_test(rands, levys)
	print t
	print prob
	print "t-test gradients levys"
	t,prob = t_test(gradients, levys)
	print t
	print prob

	plot_random_gradient_levys(rands, gradients, levys)
	"""
	#ificant p value, exactly as wanted!
	
	
	#random_base = np.load('random_walk_base.npy')
	#levy_base = np.load('levy_flight_base.npy')
	#gradient_base = np.load('gradient_base.npy')
	#plot_example_gradient_and_random(random_base, gradient_base)
	#plot_example_random_levy_gradient(random_base, levy_base, gradient_base, base=mat)
	#plt.imshow(random_base)
	#plt.show()

	#plot the changes for animation purposes
	#plot_image_changes(N=200, radius=5,plot_after=1000000, save_name='vocal_learning_development_8')
	#this is actually going to test the robustness of the method
	#for i in range(5,20):
	#	save_name = 'vocal_learning_radius_' + str(i)
	#	plot_image_changes(N=200, radius=i, plot_after=1000000, save_name=save_name)
	#	print "completed version: " + str(i)

	#I also need to simulate the noies properly. let's do that for a bit and generate the naimation ther
	#plot_image_changes(N=400, radius=5, plot_after=100000, save_name='vocal_learning_noise_0.17', multiplier=0.17)
	#print "done!"

	# If I want to do the extreme computational load, I can theoretically test the robustness in a lot of cases
	# first I just need to check how robust it is relative to the r hyperparameter,
	# which I haven't done that often. I'll really need to run a whole load of these simulations
	# to get it to work well, or use julia instead because loops in python are REALLY slow
	# and this isn't good!
	# call the animations and make them directly - it's all in the same directory
	# so this should work even though it's very hacky

	for i in range(3,20):
		osstr = 'python animate_seals.py vocal_learning_radius_' + str(i)+'.npy'
		os.system(osstr)
		print "done!"

	# things that need to be done - check the robustness of the results for r - hopefully that will come through
	# check on my claim of things for the randomness although I'm suspicious about this. hopefully at some point
	# it will slow convergence, but I'm not sure... it doesn't seem to have with small values, but perhaps closer to criticality?!
	# then what I will have to do is choose a point where gradients seem reasonably established
	# and rerun all the actual running levy flights and the like till atop
	# and see if that works or is reasonable at all
	# and hopefully it will be
	# and there are a multitude of other parameters I can test in the model
	# and so provide it with the background on too, so that should be nice
	# I forget what the base for this is! 
	# it works butI dno't thin it's the correct base - I'm going to have to rerun all the levy tests
	# which really isn't the end of the world. it doesn't take that long!
	# yeah, I don' think this randomness is working... I wonder why I got it seemingly stuck
	# that one time, having asserted this I'm now going to be in serious trouble... dagnabbit!
