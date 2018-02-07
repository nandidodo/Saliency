#okay, copy out the tensorflow gan thing here
# in the hope I have some idea what is going on, and figure it out/consolidate it
# then perhaps advance a little to get to some rejection sampling ideas tested

import tensorflow as tensorflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse

def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1./tf.sqrt(in_dim/2.)
	return xavier_stddev

def discriminator(X, reuse=False):
	with tf.variable_scope('discriminator')
		if(reuse):
			tf.get_variable_scope().reuse_variables()

	J = 784 #not sure why
	K = 128 # still no idea?
	L = 1

	W1 = tf.get_variable('D_W1', [J,K])
	B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())
	W2 = tf.get_variable('D_W2', [K,L], initializer=tf.random_normal_initializer(stddev=xavier_init([K,L])))
	B2 = tf.get_variable('D_B2', [L], initializer=tf.constant_initializer)


	#summary variables
	tf.summary.histogram('weight1',W1)
	tf.summary.histogram('weight2', W2)
	tf.summary.histogram('biases1', B1)
	tf.summary.histogram('biases2', B2)

	fc1 = tf.nn.relu((tf.matmul(X,W1) + B1))
	logits = tf.matmul(fc1, W2) + B2 #no activation function here
	probs = tf.nn.sigmoid(logits)	#seems reasonable! sigmoid to get probabilities - surprised it's not softmax???
	return probs, logits

def generator(X):
	with tf.variable_scope('generator')
		K = 128
		L = 784

		W1 = tf.get_variable('G_W1', [100,K], initializer=tf.random_normal_initializer(stddev=xavier_init([100, K])))
		B1 = tf.get_variable('G_B1', [K], initializer=tf.constant_initializer())

		W2 = tf.get_variable('G_W2', [K,L], initializer=tf.random_normal_initializer(stddev=xavier_init([K,L])))
		B2 = tf.get_variable('G_B2', [L], initializer = tf.constant_initializer())

		#summary
		tf.summary.histogram('weight1', W1)
		tf.summary.histogram('weight2', W2)
		tf.summary.histogram('biases1', B1)
		tf.summary.histogram('biases2', B2)

		fc1 = tf.nn.relu(tf.matmul(X,W1) + B1)
		fc2 = tf.matmul(fc1, W2) + B2
		prob = tf.nn.sigmoid(fcs)
		return prob


def read_data():
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
	return mnist

def plot(samples):
	plot_size = (8,8)
	fig = plt.figure(figsize=plot_size)
	gs = gridspec.GridSpec(8,8)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.setyticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28,28), cmap='Greys_r')
	return fig

def train(logdir, batch_size, epochs=10000):
	mnist = read_data()
	with tf.variable_scope('placeholder'):
		#raw image
		X = tf.placeholder(tf.float32, [None, 784])
		tf.summary.image('raw image', tf.reshape(X, [-1, 28,28,1]),3)
		#noise
		z = tf.placeholder(tf.float32, [None, 100])
		tf.summary.histogram('Noise', z)

	with tf.variable_scope('GAN'):
		G = generator(z)
		D_real, D_real_logits = discriminator(X, reuse=False)
		D_fake, D_fake_logits = discriminator(G, reuse=True)
	tf.summary.image('generated image', tf.reshape(G, [-1, 28,28,1]),3)

	with tf.variable_scope('Prediction'):
		tf.summary.histogram('real', D_real)
		tf.summary.histogram('fake', D_fake)

	with tf.variable_scope('D_loss'):
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
		d_loss =  d_loss_real + d_loss_fake

		tf.summary.scalar('d_loss_real', d_loss_real)
		tf.summary.scalar('d_loss_fake', d_loss_fake)
		tf.summary.scalar('d_loss', d_loss)

	with tf.name_scope('G_loss'):
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
		tf.summary.scalar('g_loss', g_loss)

	tvar = tf.trainable_variables()
	dvar = [var for var in tvar if 'discriminator' in var.name]
	gvar = [var for var in tvar if 'generator' in var.name]

	with tf.name_scope('train'):
		d_train_step = tf.train.AdamOptimizer().minimize(d_loss, var_list=dvar)
		g_train_step = tf.train.AdamOptimizer().minimize(g_loss, var_list=gvar)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter('tmp/mnist/'+logdir)
	writer.add_graph(sess.graph)

	num_img = 0
	if not os.path.exists('output/'):
		os.makedirs('output/')

	for i in range(epochs):
		batch_X,  = mnist.train.next_batch(batch_size)
		batch_noise =np.ranom.uniform(-1., 1., [batch_size, 100])

		if i % 500 ==0:
			samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64,100])})
			fig = plot(samples)
			plt.savefig('output/%s.png' %str(num_img).zfill(3), bbox_inches='tight')
			num_img +=1
			plt.close(fig)

		_, d_loss_print = sess.run([d_train_step, d_loss],feed_dict={X:batch_X, z_batch_noise})
		_, g_loss_print = sess.run([g_train_step, g_loss], feed_dict={z:batch_noise})

		if i %100 ==0:
			s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise})
			writer.add_summary(s,i)
			print('epoch:%d g_loss:%f, d_loss%f', %(i,g_loss_print, d_loss_print))

	if __name__ == '__main__':
		parser=argparse.ArgumentParser(description='Train vanilla Gan using fully connected layers')
		parser.add_argumant('--logdir', type=str, default="1", help='logdir for tensorboard, give a string')
		parser.add_argumant('--batch_size', type=int, default=64, help='batch size: give an int')
		args = parser.parse_args()

		train(logdir=args.logdir, batch_size=args.batch_size)