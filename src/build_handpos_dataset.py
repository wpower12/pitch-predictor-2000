from data.load_data import *
import pickle

years  = [2013, 2014, 2015, 2016]
months = [3, 4, 5, 6, 7, 8, 9, 10]

fn = "full_handpos_pitches"

X, f, y, ls = build_data_set_from_years(years, months)

print(len(X))

pickle.dump([X, f, y], open("../data/{}.p".format(fn), "wb"))
