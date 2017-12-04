import sys
sys.path.append('../')
from models.WindowModel import WindowModel 
from data.load_data import *  

# model_type    = {basic, features} // the data does or does not include some feature vector for each sequence
# learning_rate = {0.01, 0.1, 1, 10 ...}
# cell_type     = {RNN, LSTM}
# layer_size    = {10, 50, 100 ...}

# x, f, y, ls = build_data_set_from_year(2016, [3,4,5])

PREFIX = "testingwindow"

NUM_FEATURES = 32
NUM_CLASSES = 16
BATCH_SIZE  = 100

x, y = build_window_data_set([2016], [3,4,5], 2)

a = WindowModel(0.01, [64, 32])
a.train([x[:2000], y[:2000]], BATCH_SIZE, 5, PREFIX)
correct = a.test([x[:100], y[:100]], PREFIX)

print(correct)