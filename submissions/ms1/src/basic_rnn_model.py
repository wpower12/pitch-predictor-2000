import pickle
import numpy as np
import tensorflow as tf 

#### Data ##########################################################################

# First we get the complete data set 
# Right I just have the first 6 months of the 2016 season.  
# Each entry here is an array with two entries.  the first contains a simple feature
# vector: [pitcher_hand, batter_hand, batter_pos],the second entry is a list of
# pitch identifiers. Each identifier by a 2 letter code, each mapping to the 
# typical pitch name.  
full_data = [] 
year = 2016
for m in [3,4,5,6,7,8]:
    fn = "../data/pitches_{}_{}.p".format(year, m)
    seqs = pickle.load(open(fn, "rb"))
    full_data += seqs

# The MLBgame api documentation is incomplete, but from reading 
# the source code, there should be a total of 16 pitch types.   
cleaned_data = [] # no 0 or 1 length sequences. 
longest_seq = 0
empties_or_single = 0
pitch_types = set()
for line in full_data:
    if(len(line[1]) > longest_seq): longest_seq = len(line[1])
    if(len(line[1]) <= 1): 
        empties_or_single += 1
    else:
        cleaned_data.append(line[1])
        for p in line[1]:# the seq is the second element, first is the feature vector
            pitch_types.add(p)

print("longest sequence length: {}\nempties: {}\ntotal (clean): {}\npitch types: {}".format(longest_seq, 
                                                                                            empties_or_single,
                                                                                            len(cleaned_data),
                                                                                            len(pitch_types)))
# Creating X - padded sequences of one-hots. Need a dictionary of pitch types to an index.
pitch_map = {
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

MAX_LENGTH = longest_seq

def create_onehot(seq):
    ret = []
    i = 0
    for p in seq:
        p_oh = np.zeros((len(pitch_map),), dtype=np.float32)
        p_oh[pitch_map[p]] = 1.0
        ret.append(p_oh)
        i += 1
    for j in range(i, MAX_LENGTH):# Pad to length. 
        ret.append(np.zeros((len(pitch_map),), dtype=np.float32))
    return ret

def create_target(seq):
    ret = []
    i = 0
    for p in seq[1:]:
        ret.append(pitch_map[p])
        i += 1
    for j in range(i, MAX_LENGTH):
        ret.append(0)
    return ret
# Actually creating x/y
X_full = [] # Sequences of onehots.
y_full = [] # index of correct pitch in the one-hot, starting at X[1]
for line in cleaned_data:
    X_full.append(create_onehot(line))
    y_full.append(create_target(line))

##### Constructing Graph ####################################################################
tf.reset_default_graph()
NUM_INPUTS  = 16    # Size of the input vector (the number of possible pitch types)
NUM_OUTPUTS = 16    # Want a pitch type out, so same size as input.
NUM_NEURONS = 10     # Number of neurons inside the RNN cell.  
MAX_SIZE    = 18    # the maximum size of a sequence.  Everything gets padded to this, and masked.
BATCH_SIZE = 5
LEARNING_RATE = 0.015

### RNN Graph
X = tf.placeholder( tf.float32, [BATCH_SIZE, MAX_SIZE, NUM_INPUTS] ) 
# y is X shifted to the left, but also converted to the *index* of the correct logit - for seq2seq loss.
y = tf.placeholder( tf.int32, [BATCH_SIZE, MAX_SIZE] ) 

# Get a 1D Tensor to hold the 'true' length of each padded sequence in a batch
collapsed_features = tf.sign(tf.reduce_max(tf.abs(X), 2)) # use max+abs to see what elements arent 0-vectors
seq_len  = tf.cast( tf.reduce_sum(collapsed_features, 1), tf.int32 ) # Count the 1's to get length.
seq_mask = tf.sequence_mask(seq_len, maxlen=MAX_SIZE, dtype=tf.float32) # Create a mask from these lengths

basic_cell   = tf.contrib.rnn.BasicRNNCell( num_units=NUM_NEURONS )
# output is shaped [BATCH_SIZE, MAX_LENGTH, NUM_NEURONS]
outputs, states = tf.nn.dynamic_rnn( basic_cell, X, dtype=tf.float32, sequence_length=seq_len ) 

### Loss, Optimization, Training.  

# seq2seq loss gets the loss by comparing the logits of the prediction 
# to the index of the correct label, given by y.  seq_mask is used to stop
# unrolling the dynamic RNN at the correct spot in the padded sequence.  
# NOTE: I think the error is here. Since the outputs are [BATCH_SIZE, MAX_LENGTH, NUM_NEURONS], they can't be
#       directly used as a prediction.  Need to turn the output into a prediction by using another
#       network layer that converts the [BATCH_SIZE, MAX_LENGTH, NUM_NEURONS] tensor into a 
#       [BATCH_SIZE, MAX_LENGTH, NUM_OUTPUTS] vector (logits for each pitch type.)
# 
# Atleast, I think this is normally done with a fully connected layer between the outputs and the inputs to the
# actual loss function. 
logits = tf.contrib.layers.fully_connected(outputs, NUM_OUTPUTS)
loss = tf.contrib.seq2seq.sequence_loss(logits, 
                                        y, 
                                        seq_mask, 
                                        average_across_timesteps=True, 
                                        average_across_batch=True)
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
training_op = optimizer.minimize( loss )
init = tf.global_variables_initializer()

#### Training ############################################################################
EPOCHS     = 10 # Will need to figure out what this should be. dVC stuff?
ITERATIONS = 10000

# NOTE: Hacky right now, but just want to get data into the model.
# TODO: actually turn the data into tf.Dataset objects? 
# TODO: or use some of the batch operations?
def get_training_batch(X, y, batch_size):
    ids = np.random.randint(0, len(X), batch_size)
    return np.array(X)[ids], np.array(y)[ids]

merged = tf.summary.merge_all()

# Testing for now, want to see if it actually updates on one batch.
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('../data', sess.graph)
    init.run()
    
    # For debugging
    X_batch, y_batch = get_training_batch( X_full, y_full, BATCH_SIZE )
    print("shape(seq_len): ", sess.run(seq_len, feed_dict={X: X_batch, y: y_batch}).shape)
    print("shape(seq_mask): ",sess.run(seq_mask, feed_dict={X: X_batch, y: y_batch}).shape)
    print("shape(outputs): ", sess.run(outputs, feed_dict={X: X_batch, y: y_batch}).shape)
    print("shape(state): ",   sess.run(states, feed_dict={X: X_batch, y: y_batch}).shape)
    print("shape(logits): ",  sess.run(logits, feed_dict={X: X_batch, y: y_batch}).shape)
    
    # TODO: Implement correct batching. 
    for i in range(ITERATIONS):
        X_batch, y_batch = get_training_batch( X_full, y_full, BATCH_SIZE )   
        _, l, summary = sess.run([training_op, loss, merged], feed_dict={X: X_batch, y: y_batch})
        summary_writer.add_summary(summary, i)
        if i%10 == 0: print("loss at i {}: {}".format(i, l))
            
    summary_writer.close()