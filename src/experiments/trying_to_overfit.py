"""
	Attempting to overfit. Need to compare a very over engineered networks in/out of sample
	accuracies. 

	The sequence model will get another layer, and more cells. The window is hard coded at 
	just two layers, so it'll just be given more cells. 

	Setting aside a validation set, the models will be trained, then the test method will be
	called on both the training data and the validation data and compared. 
"""
import pickle, random, sys
import numpy as np
sys.path.append('../')
from models.SequenceModel import *
from models.WindowModel import *

RATE = 0.01
EPOCHS = 5
BATCH_SIZE = 50
SEQ_LAYERSIZES = [128, 64, 32]
WIN_LAYERSIZES = [128, 64]
NUM_TRAIN = 20000
NUM_TEST  = 1000

seq_fn = "../../data/full_handposbase_pitches.p"
win_fn = "../../data/full_window_2_hbp_pitches.p"

X_s, f_s, y_s = pickle.load(open(seq_fn, "rb"))

X_train, f_train, y_train = X_s[:NUM_TRAIN], f_s[:NUM_TRAIN], y_s[:NUM_TRAIN]
X_test = X_s[NUM_TRAIN:NUM_TRAIN+NUM_TEST] 
f_test = f_s[NUM_TRAIN:NUM_TRAIN+NUM_TEST] 
y_test = y_s[NUM_TRAIN:NUM_TRAIN+NUM_TEST]

id_str  = "seq_overfitting_test"
s_model = SequenceModel("feature", RATE, "lstm", SEQ_LAYERSIZES)
s_model.train([X_train, f_train, y_train], BATCH_SIZE, EPOCHS, id_str)
s_acc_in  = s_model.test([X_train, f_train, y_train], id_str)
s_acc_out = s_model.test([X_test, f_test, y_test], id_str)

print("seq acc_in: {} acc_out: {}".format(s_acc_in, s_acc_out))

X_w, y_w = pickle.load(open(win_fn, "rb"))

X_train, y_train = X_w[:NUM_TRAIN], y_w[:NUM_TRAIN]
X_test = X_w[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
y_test = y_w[NUM_TRAIN:NUM_TRAIN+NUM_TEST]

id_str = "win_overfitting_test"
w_model = WindowModel(RATE, WIN_LAYERSIZES)
w_model.train([X_train, y_train], BATCH_SIZE, EPOCHS, id_str)
w_acc_in  = w_model.test([X_train, y_train], id_str)
w_acc_out = w_model.test([X_test, y_test], id_str) 

print("win acc_in: {} acc_out: {}".format(w_acc_in, w_acc_out))