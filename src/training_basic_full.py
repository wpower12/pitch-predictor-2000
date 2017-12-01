"""
Full training of the basic model. Using parameters found from experimentation. 
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

NUM_TRAIN = 450000
NUM_TEST  = 10000

X_train, f_train, y_train = X_full[:NUM_TRAIN], f_full[:NUM_TRAIN], y_full[:NUM_TRAIN]
X_test, f_test, y_test    = X_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST], f_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST], y_full[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
print("data split. {} training examples, {} test examples.".format(NUM_TRAIN, NUM_TEST))

RATE = 0.1
LAYERSIZE = 10
EPOCHS = 5
BATCH_SIZE = 100
TEST_ID = "fullbasiclstm01"

model_type = "basic"
cell_type  = "lstm"

r_str  = str(RATE).replace(".", "-")
id_str = "{}_{}_{}_{}_{}".format(TEST_ID, model_type, cell_type, r_str, str(LAYERSIZE))

print("training model: {}".format(id_str))
model = PitchModel(model_type, RATE, cell_type, LAYERSIZE)
model.train([X_train, f_train, y_train], BATCH_SIZE, EPOCHS, id_str)
acc = model.test([X_test, f_test, y_test], id_str)
print("acc: {}".format(acc))							




