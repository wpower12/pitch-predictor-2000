from data.load_data import *
import pickle

years  = [2016, 2015, 2014, 2013]
months = [3, 4, 5, 6, 7, 8, 9, 10]

fn = "full_handposbase_pitches"

# The load_data functions are responsible for converting the 'human readable' list of records 
# into actual np arrays that can be properly digested by tensorflow.  

# they combine multiple year and months of partial data into one large data set.  
# I dump this to a file to make the trianing files cleaner. 

X, f, y, ls = build_data_set_from_years(years, months)

print(len(X))

print(X[0], f[0])

pickle.dump([X, f, y], open("../data/{}.p".format(fn), "wb"))
