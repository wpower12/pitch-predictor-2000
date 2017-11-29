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

class PitchModel:
	PITCH_MAP = {
	    'KC': 0,
	    'CH': 1,
	    'SL': 2,
	    'SI': 3,
	    'FO': 4,
	    'FS': 5,
	    'CU': 6,
	    'PO': 7,
	    'KN': 8,
	    'FF': 9,
	    'EP': 10,
	    'IN': 11,
	    'SC': 12,
	    'FT': 13,
	    'FC': 14,
	    'UN': 15
	}

	POS_MAP = {
	    '1B': 0,
	    '2B': 1,
	    '3B': 2,
	    'PR': 3,
	    'P':  4,
	    'C':  5,
	    'DH': 6,
	    'SS': 7,
	    'PH': 8,
	    'CF': 9,
	    'RF': 10,
	    'LF': 11
	}

	HAND_MAP ={
	    'L': 0.0,
	    'R': 1.0,
	    'B': 0.5, # Just doing this for now. 
	    'S': 0.5
	}

	def __init__(self, model_type, learning_rate, cell_type, layer_size):
		self.MODEL_TYPE    = model_type
		self.LEARNING_RATE = learning_rate
		self.CELL_TYPE     = cell_type 
		self.LAYER_SIZE    = layer_size 

	def run(self, data, batch_size, epochs, file_prefix):
		# data = [x, f, y]
		tf.reset_default_graph()
		NUM_INPUTS   = 16    # Size of the input vector (the number of possible pitch types)
		NUM_OUTPUTS  = 16    # Want a pitch type out, so same size as input.
		MAX_SIZE     = len(data[0][0])    # the maximum size of a sequence.  Everything gets padded to this, and masked.
		
		if self.MODEL_TYPE == "feature":
			FEATURE_SIZE = len(data[1][0])    # Size of the additional feature vector.
			F = tf.placeholder(tf.float32, [None, FEATURE_SIZE], name="F") 
		else:
			F = tf.placeholder(tf.float32, [batch_size, FEATURE_SIZE], name="F") 

		X = tf.placeholder(tf.float32, [None, MAX_SIZE, NUM_INPUTS], name="X")
		y = tf.placeholder(tf.int32, [None, MAX_SIZE], name="y") 

		# Some operations need sequence length and masks for each example.
		collapsed_elems = tf.sign(tf.reduce_max(tf.abs(X), 2)) # use max+abs to see what elements arent 0-vectors
		seq_len  = tf.cast( tf.reduce_sum(collapsed_elems, 1), tf.int32, name="seq_len" ) # Count the 1's to get length.
		seq_mask = tf.sequence_mask(seq_len, maxlen=MAX_SIZE, dtype=tf.float32, name="seq_mask") # Create a mask from these lengths

		if self.CELL_TYPE == "rnn":
			cell = tf.contrib.rnn.BasicRNNCell(num_units=self.LAYER_SIZE)
		else:
			cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.LAYER_SIZE)

		outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_len) 

		if self.MODEL_TYPE == "feature":
			F_expanded = tf.tile(tf.expand_dims(F, 1), [1, MAX_SIZE, 1])
			combined_outputs = tf.concat((outputs, F_expanded), 2)
			logits = tf.contrib.layers.fully_connected(combined_outputs, NUM_OUTPUTS)
		else:
			logits = tf.contrib.layers.fully_connected(outputs, NUM_OUTPUTS)

		prediction = tf.argmax(logits, axis=1, name="prediction", output_type=tf.int32)

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
		with tf.Session() as sess:
			summary_writer = tf.summary.FileWriter('../graphs', sess.graph)
			init.run()
			for i in range(iterations):
				X_batch, F_batch, y_batch = get_training_batch(data[0], data[1], data[2], batch_size)
				_, l, summary = sess.run([training_op, loss, merged], feed_dict={X: X_batch, F: F_batch, y: y_batch})
				summary_writer.add_summary(summary, i)
				if i%10 == 0: print("loss at i {}: {}".format(i, l))

			fpath = saver.save(sess, "../graphs/{}.ckpt".format(file_prefix))
			print("model saved as: {}".format(fpath))
			summary_writer.close()

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
