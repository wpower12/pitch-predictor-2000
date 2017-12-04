""" WARNING - This makes a huge file. 1.5~ish gigs."""

import pickle, sys
from load_data import *

years  = [2016, 2015, 2014, 2013]
months = [3, 4, 5, 6, 7, 8, 9, 10]

fn = "full_window_2_hbp_pitches"

window_size = 2
X, y = build_window_data_set(years, months, window_size)

print(len(X))
print(X[0])

pickle.dump([X, y], open("../data/{}.p".format(fn), "wb"))
