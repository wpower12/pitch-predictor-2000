""" Window Model

	A two layer fully connected neural network. The input represents a window of
	pitches, and the output corresponds to a pitch in the pitch_map.  The output
	layer is used to evaluate the loss with soft_max_cross_entropy between the 
	output and the label.  The label is a one-hot encoding of the pitch type that 
	follows those in the window. 

	The model does not need any information about the size of the window, or even 
	the size of the pitch_map, as the information is inffered from the provided 
	data. 
"""
import tensorflow as tf
import numpy as np
from progress.bar import Bar

class WindowModel:
	def __init__(self, learning_rate, layer_sizes):
		self.LEARNING_RATE = learning_rate
		self.LAYER_SIZES    = layer_sizes 

	""" Trains a model on the data using the stored parameters. Saves the model to the relevant tf files, using the file_prefix."""
	def train(self, data, batch_size, epochs, file_prefix):
		tf.reset_default_graph()
		NUM_INPUTS   = len(data[0][0])  # Size of the input vector 2*num_pitches+num_additional_features
		NUM_OUTPUTS  = len(data[1][0])	# size of output: num_pitches       

		X = tf.placeholder(tf.float32, [None, NUM_INPUTS], name="X")
		y = tf.placeholder(tf.int32, [None, NUM_OUTPUTS], name="y") 

		fc1    = tf.contrib.layers.fully_connected(X,    self.LAYER_SIZES[0])
		fc2    = tf.contrib.layers.fully_connected(fc1,  self.LAYER_SIZES[1])
		logits = tf.contrib.layers.fully_connected(fc2, NUM_OUTPUTS)

		# TODO - loss function? hinge loss?  fuck. 
		loss = tf.losses.softmax_cross_entropy(y, logits)
		prediction = tf.argmax(logits, axis=1, name="prediction", output_type=tf.int32)
		tf.summary.scalar('{}_loss'.format(file_prefix), loss)

		optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
		training_op = optimizer.minimize(loss)
		init = tf.global_variables_initializer()

		#### Training Phase ###############
		iterations_per_epoch = int((1.0*len(data[0]))/(1.0*batch_size))

		# Random selection of batch
		def get_training_batch(X, y, i, batch_size):
		    ids = np.random.randint(0, len(X), batch_size)
		    l = i*batch_size
		    if l < len(X)-batch_size:
		    	u = l + batch_size
		    else:
		    	u = len(X)
		    return np.array(X)[l:u], np.array(y)[l:u]

		saver = tf.train.Saver()
		merged = tf.summary.merge_all()	
		p_bar = Bar('batches', max=iterations_per_epoch*epochs)
		with tf.Session() as sess:
			summary_writer = tf.summary.FileWriter('../graphs', sess.graph)
			init.run()
			for e in range(epochs):
				for i in range(iterations_per_epoch):
					X_batch, y_batch = get_training_batch(data[0], data[1], i, batch_size)
					_, summary = sess.run([training_op, merged], feed_dict={X: X_batch, y: y_batch})
					summary_writer.add_summary(summary, i)
					p_bar.next()

			p_bar.finish()
			fpath = saver.save(sess, "../graphs/{}.ckpt".format(file_prefix))
			print("model saved as: {}".format(fpath))
			summary_writer.close()
	
	""" Loads a model from the tf files indicated by file_prefix, then calculates the number of correct predictions given the data."""		
	def test(self, data, file_prefix):
		tf.reset_default_graph()
		sess = tf.Session()
		
		saver = tf.train.import_meta_graph('../graphs/{}.ckpt.meta'.format(file_prefix))
		saver.restore(sess, tf.train.latest_checkpoint('../graphs/'))

		graph = tf.get_default_graph()

		prediction = graph.get_tensor_by_name("prediction:0")
		X     = graph.get_tensor_by_name("X:0")
		y     = graph.get_tensor_by_name("y:0")

		correct = tf.count_nonzero(tf.equal(prediction, tf.argmax(y, axis=1, output_type=tf.int32)))

		feeddict = {X: data[0], y: data[1]}
		c = sess.run([correct], feeddict)
		return c
