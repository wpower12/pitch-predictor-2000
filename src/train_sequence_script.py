"""
Script used to simplify training of models. 
"""
import pickle
import numpy as np
import random
import sys
from models.SequenceModel import *

# Defaults for all the parameters
FULL_DATA_FN = "../data/full_handposbase_pitches.p"
model_type = "feature"
cell_type  = "lstm"
RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 50
TEST_ID = "multilayer_001_64_32"
LAYERSIZES = [64, 32]
NUM_TRAIN = 10000
NUM_TEST  = 4500

if len(sys.argv) > 1:
	FULL_DATA_FN = sys.argv[1]
	model_type   = sys.argv[2]
	cell_type    = sys.argv[3]
	RATE         = float(sys.argv[4])
	EPOCHS       = int(sys.argv[5])
	BATCH_SIZE   = int(sys.argv[6])
	NUM_TRAIN 	 = int(sys.argv[7])
	NUM_TEST  	 = int(sys.argv[8])
	TEST_ID      = sys.argv[9]	# Used to identify the files that 'save' the model and its parameters.
	LAYERSIZES   = [int(l) for l in sys.argv[10:]]

print("model type: {}\ncell type: {}\nlayers: {}\n".format(model_type,
														cell_type,
														str(LAYERSIZES)))

X_full, f_full, y_full = pickle.load(open(FULL_DATA_FN, "rb"))
print("data loaded from {}".format(FULL_DATA_FN))

data = list(zip(X_full, f_full, y_full))
random.shuffle(data)
X_full, f_full, y_full = zip(*data)

X_train, f_train, y_train = X_full[:NUM_TRAIN], f_full[:NUM_TRAIN], y_full[:NUM_TRAIN]
X_test, f_test, y_test    = X_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST], f_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST], y_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
print("data split. {} training examples, {} test examples.".format(NUM_TRAIN, NUM_TEST))

r_str  = str(RATE).replace(".", "-")
id_str = "{}_{}_{}".format(TEST_ID, model_type, cell_type)

print("training model: {}".format(id_str))
model = SequenceModel(model_type, RATE, cell_type, LAYERSIZES)
model.train([X_train, f_train, y_train], BATCH_SIZE, EPOCHS, id_str)

# The sequence model test method ouputs a %
acc_out = model.test([X_test, f_test, y_test], id_str)
acc_in  = model.test([X_train, f_train, y_train], id_str)

print("acc_in: {}\nacc_out: {}".format(acc_in, acc_out))							




