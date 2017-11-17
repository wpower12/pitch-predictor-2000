import pickle, sys, mlbgame, pymysql

PITCH_DATA_DIR = "data/"
TEST_DATA 	   = "pitches_2016_3.p"

seqs = pickle.load(open(PITCH_DATA_DIR+TEST_DATA, "rb"))

for p in seqs[:100]:
	print(p)