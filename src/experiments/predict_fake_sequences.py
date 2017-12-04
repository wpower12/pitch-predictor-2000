"""
Testing if the recurrent network can atleast predict a simple sequence. 
"""
import pickle
import numpy as np
import random
import sys
sys.path.append('../')
from models.SequenceModel import *
from data.load_data import *

model_type = "basic"
cell_type  = "lstm"
RATE = 0.01
EPOCHS = 5
BATCH_SIZE = 50
LAYERSIZES = [64, 32]

id_str = "testing_fake_data"

# (length, alpha, period, reps):
alpha = ['a', 'b', 'c', 'd']
alpha_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
seqs = build_fake_sequences(7000, alpha, 10, 4)

# seqs, alpha_map, max_len
X_full, f_full, y_full = build_fake_data_set(seqs, alpha_map, 7*4)

X_train, f_train, y_train = X_full[:6000], f_full[:6000], y_full[:6000]
X_test, f_test, y_test = X_full[6000:], f_full[6000:], y_full[6000:]

s_model = SequenceModel(model_type, RATE, cell_type, LAYERSIZES)
s_model.train([X_train, f_train, y_train], BATCH_SIZE, EPOCHS, id_str)
s_acc = s_model.test([X_test, f_test, y_test], id_str)

print("seq acc: {}".format(s_acc))							




