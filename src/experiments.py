"""
Need to experiment with the possible values for model hyperparameters on smaller validation folds.

model_type    = {basic, features} // 2 - Trying to find best for each.

cell_type     = {RNN, LSTM}       // 2
learning_rate = {0.01, 0.1, 1}    // 4
layer_size    = {10, 20, 40}      // 3
								    ---
								     24 Total

For basic and feature, we want to find the parameter value that does best on the small fold.
The final comparison between the basic/feature models will be between models trained on the 
full data set, using the hyperparameter values found in this experiment.

The dataset is about 500k strong right now.  I need to tuck away a portion as a test set and never
look at it.  Is 10% enough? 50k pitch sequences? How about 5%? Thats just 25k, leaves a lot of 
data left over to make the 24 folds.  
"""
import pickle
import numpy as np
import random
from models.PitchModel import *

FULL_DATA_FN = "../data/full_handpos_pitches.p"
X_full, f_full, y_full = pickle.load(open(FULL_DATA_FN, "rb"))
print("data loaded from {}".format(FULL_DATA_FN))

data = list(zip(X_full, f_full, y_full))
random.shuffle(data)
X_full, f_full, y_full = zip(*data)

NUM_TRAIN = 50000
NUM_TEST  = 5000

X_train, f_train, y_train = X_full[:NUM_TRAIN], f_full[:NUM_TRAIN], y_full[:NUM_TRAIN]
X_test, f_test, y_test    = X_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST], f_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST], y_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
print("data split. {} training examples, {} test examples.".format(NUM_TRAIN, NUM_TEST))

RATES = [0.01, 0.1, 1]
LAYERSIZES = [10, 20, 40]
EPOCHS = 3
BATCH_SIZE = 100
TEST_ID = "test01"

# Model(model_type, learning_rate, cell_type, layer_size)
# model.train(data, batch_size, epochs, prefix_id_for_experiment )
# model.test(data, prefix_id_for_experiment)

for rate in RATES:
	for size in LAYERSIZES:
		for model_type in ["feature", "basic"]:
			for cell_type in ["rnn", "lstm"]:
				model = PitchModel(model_type, rate, cell_type, size)
				r_str  = str(rate).replace(".", "-")
				id_str = "{}_{}_{}_{}_{}".format(TEST_ID, model_type, cell_type, r_str, str(size))
				print("training model: {}".format(id_str))
				model.train([X_train, f_train, y_train], BATCH_SIZE, EPOCHS, id_str)
				acc = model.test([X_test, f_test, y_test], id_str)
				print("acc: {}".format(acc))



