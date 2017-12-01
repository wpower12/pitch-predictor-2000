""" PitchModel.py
This is the general class for the model. It's being split out into a class
in this way to help simplify the experimentation. Would like to be able
to create a single class that can be configured to run the different 
types of things I want to experiment on.

model =  Model(model_type, learning_rate, cell_type, layer_size)

model_type    = {basic, features} // the data does or does not include some feature vector for each sequence
learning_rate = {0.01, 0.1, 1, 10 ...}
cell_type     = {RNN, LSTM}
layer_size    = {10, 50, 100 ...}

model.run(data, batch_size, epochs, prefix_id_for_experiment )
model.test(data, prefix_id_for_experiment)

data will be a 3 entry array: [X, F, y] 
"""
import tensorflow as tf
import numpy as np
from progress.bar import Bar

class PitchModel:
	def __init__(self, model_type, learning_rate, cell_type, layer_sizes):
		self.MODEL_TYPE    = model_type
		self.LEARNING_RATE = learning_rate
		self.CELL_TYPE     = cell_type 
		self.LAYER_SIZES    = layer_sizes 

	def train(self, data, batch_size, epochs, file_prefix):
		# data = [x, f, y]
		tf.reset_default_graph()
		NUM_INPUTS   = 16    # Size of the input vector (the number of possible pitch types)
		NUM_OUTPUTS  = 16    # Want a pitch type out, so same size as input.
		MAX_SIZE     = len(data[0][0])    # the maximum size of a sequence.  Everything gets padded to this, and masked.
		
		if self.MODEL_TYPE == "feature":
			FEATURE_SIZE = len(data[1][0])    # Size of the additional feature vector.
			F = tf.placeholder(tf.float32, [None, FEATURE_SIZE], name="F") 
		else:
			F = tf.placeholder(tf.float32, [None, None], name="F") 

		X = tf.placeholder(tf.float32, [None, MAX_SIZE, NUM_INPUTS], name="X")
		y = tf.placeholder(tf.int32, [None, MAX_SIZE], name="y") 

		# Some operations need sequence length and masks for each example.
		collapsed_elems = tf.sign(tf.reduce_max(tf.abs(X), 2)) # use max+abs to see what elements arent 0-vectors
		seq_len  = tf.cast( tf.reduce_sum(collapsed_elems, 1), tf.int32, name="seq_len" ) # Count the 1's to get length.
		seq_mask = tf.sequence_mask(seq_len, maxlen=MAX_SIZE, dtype=tf.float32, name="seq_mask") # Create a mask from these lengths

		if self.CELL_TYPE == "rnn":
			layers = [tf.contrib.rnn.BasicRNNCell(num_units=s) for s in self.LAYER_SIZES]
		else:
			layers = [tf.contrib.rnn.BasicLSTMCell(num_units=s) for s in self.LAYER_SIZES]

		if len(layers) > 1:
			outputs, states = tf.nn.dynamic_rnn(layers[0], X, dtype=tf.float32, sequence_length=seq_len) 
		else:
			outputs, states = tf.nn.rnn_cell.MultiRNNCell(layers)

		if self.MODEL_TYPE == "feature":
			F_expanded = tf.tile(tf.expand_dims(F, 1), [1, MAX_SIZE, 1])
			combined_outputs = tf.concat((outputs, F_expanded), 2)
			logits = tf.contrib.layers.fully_connected(combined_outputs, NUM_OUTPUTS)
		else:
			logits = tf.contrib.layers.fully_connected(outputs, NUM_OUTPUTS)

		prediction = tf.argmax(logits, axis=2, name="prediction", output_type=tf.int32)

		### Loss, Optimization, Training.  
		loss = tf.contrib.seq2seq.sequence_loss(logits, 
		                                        y, 
		                                        seq_mask, 
		                                        average_across_timesteps=True, 
		                                        average_across_batch=True)
		tf.summary.scalar('{}_loss'.format(file_prefix), loss)	
		optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
		training_op = optimizer.minimize(loss)
		init = tf.global_variables_initializer()

		#### Training Phase ###############
		iterations = int((1.0*epochs*len(data[0]))/(1.0*batch_size))

		# Random selection of batch
		def get_training_batch(X, f, y, batch_size):
		    ids = np.random.randint(0, len(X), batch_size) 
		    return np.array(X)[ids], np.array(f)[ids], np.array(y)[ids]

		merged = tf.summary.merge_all()
		saver = tf.train.Saver()
		p_bar = Bar('training', max=iterations)
		with tf.Session() as sess:
			summary_writer = tf.summary.FileWriter('../graphs', sess.graph)
			init.run()
			for i in range(iterations):
				X_batch, F_batch, y_batch = get_training_batch(data[0], data[1], data[2], batch_size)
				_, l, summary = sess.run([training_op, loss, merged], feed_dict={X: X_batch, F: F_batch, y: y_batch})
				summary_writer.add_summary(summary, i)
				p_bar.next()

			p_bar.finish()
			fpath = saver.save(sess, "../graphs/{}.ckpt".format(file_prefix))
			print("model saved as: {}".format(fpath))
			summary_writer.close()
			X_batch, F_batch, y_batch = get_training_batch(data[0], data[1], data[2], batch_size)
			
	def test(self, data, file_prefix):
		tf.reset_default_graph()
		sess = tf.Session()
		
		saver = tf.train.import_meta_graph('../graphs/{}.ckpt.meta'.format(file_prefix))
		saver.restore(sess, tf.train.latest_checkpoint('../graphs/'))

		graph = tf.get_default_graph()

		prediction = graph.get_tensor_by_name("prediction:0")
		X     = graph.get_tensor_by_name("X:0")
		F     = graph.get_tensor_by_name("F:0")
		y     = graph.get_tensor_by_name("y:0")

		# seq_mask = graph.get_tensor_by_name("seq_mask:0")
		seq_len  = graph.get_tensor_by_name("seq_len:0")
		seq_mask = tf.sequence_mask(seq_len, maxlen=len(data[0][0]), dtype=tf.bool)

		correct = tf.count_nonzero(tf.logical_and(tf.equal(prediction, y), seq_mask))
		total = tf.count_nonzero(seq_mask)

		feeddict = {X: data[0], F: data[1], y: data[2]}
		c, t = sess.run([correct, total], feeddict)
		return (1.0*c)/(1.0*t)
