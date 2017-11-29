from models.PitchModel import PitchModel 
from data.load_data import *  

# model_type    = {basic, features} // the data does or does not include some feature vector for each sequence
# learning_rate = {0.01, 0.1, 1, 10 ...}
# cell_type     = {RNN, LSTM}
# layer_size    = {10, 50, 100 ...}

x, f, y, ls = build_data_set_from_year(2016, [3,4,5])

PREFIX = "test_feature_lstm_00"
batch_size = 10
epochs = 1

a = PitchModel("feature", 0.01, "lstm", 10)

# [x[:100], f[:100], y[:100]]
# [x, f, y]

a.run([x[:100], f[:100], y[:100]], batch_size, epochs, PREFIX)
acc = a.test([x[:100], f[:100], y[:100]], PREFIX)
print("model: {}\nacc: {}".format(PREFIX, acc))