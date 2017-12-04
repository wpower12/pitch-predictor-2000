"""
Script used to simplify training of models. 
"""
import pickle
import numpy as np
import random
import sys
from models.WindowModel import *

# Defaults for all the parameters
FULL_DATA_FN = "../data/full_window_2_hbp_pitches.p"
RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 50
TEST_ID = "window_001_64_32"
LAYERSIZES = [64, 32]
NUM_TRAIN = 10000
NUM_TEST  = 4500

if len(sys.argv) > 1:
	FULL_DATA_FN = sys.argv[1]
	RATE         = float(sys.argv[2])
	EPOCHS       = int(sys.argv[3])
	BATCH_SIZE   = int(sys.argv[4])
	NUM_TRAIN 	 = int(sys.argv[5])
	NUM_TEST  	 = int(sys.argv[6])
	TEST_ID      = sys.argv[7]	# Used to identify the files that 'save' the model and its parameters.
	LAYERSIZES   = [int(l) for l in sys.argv[8:]] # should only be 2 values.

print("window model: layers: {}".format(str(LAYERSIZES)))

X_full, y_full = pickle.load(open(FULL_DATA_FN, "rb"))
print("data loaded from {}".format(FULL_DATA_FN))

data = list(zip(X_full, y_full))
random.shuffle(data)
X_full, y_full = zip(*data)

X_train, y_train = X_full[:NUM_TRAIN], y_full[:NUM_TRAIN]
X_test = X_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
y_test = y_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
print("data split. {} training examples, {} test examples.".format(NUM_TRAIN, NUM_TEST))

print("training model: {}".format(TEST_ID))
model = WindowModel(RATE, LAYERSIZES)
model.train([X_train, y_train], BATCH_SIZE, EPOCHS, TEST_ID)

# window model outputs a count of the correct predictions. 
acc_out = float(model.test([X_test, y_test], TEST_ID)[0])/float(NUM_TEST)
acc_in  = float(model.test([X_train, y_train], TEST_ID)[0])/float(NUM_TRAIN)

print("acc_in: {}\nacc_out: {}".format(acc_in, acc_out))							




