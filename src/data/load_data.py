import pickle 
import numpy as np

PITCH_MAP = {
    'KC': 0,
    'CH': 1,
    'SL': 2,
    'SI': 3,
    'FO': 4,
    'FS': 5,
    'CU': 6,
    'PO': 7,
    'KN': 8,
    'FF': 9,
    'EP': 10,
    'IN': 11,
    'SC': 12,
    'FT': 13,
    'FC': 14,
    'UN': 15
}
POS_MAP = {
    '1B': 0,
    '2B': 1,
    '3B': 2,
    'PR': 3,
    'P':  4,
    'C':  5,
    'DH': 6,
    'SS': 7,
    'PH': 8,
    'CF': 9,
    'RF': 10,
    'LF': 11
}
HAND_MAP ={
    'L': 0.0,
    'R': 1.0,
    'B': 0.5, # Just doing this for now. 
    'S': 0.5
}
## Helper Functions ##############################

# creates the hand+pos feature vector for a sequence.
def create_handpos_feature_vec(seq):
    # Create positions one-hot
    pos = np.zeros((len(POS_MAP),), dtype=np.float32)
    for p in seq[0][2].split("-"):
        pos[POS_MAP[p]] = 1.0
    
    # Handedness features
    hands = np.zeros((2,), dtype=np.float32)
    hands[0] = HAND_MAP[seq[0][0]]
    hands[1] = HAND_MAP[seq[0][1]]
    return np.concatenate((hands, pos))
    
# Creates padded one-hot sequences.
def create_oneshot_seq(seq, max_len):
    pitches = []
    i = 0
    for p in seq[1]:
        p_oh = np.zeros((len(PITCH_MAP),), dtype=np.float32)
        p_oh[PITCH_MAP[p]] = 1.0
        pitches.append(p_oh)
        i += 1
    for j in range(i, max_len):# Pad to length. 
        pitches.append(np.zeros((len(PITCH_MAP)), dtype=np.float32))
    return np.array(pitches)

# Creates a target tensor. Value = index of correct next pitch in pitch vector
def create_target(seq, max_len):
    ret = []
    i = 0
    for p in seq[1][1:]:
        ret.append(PITCH_MAP[p])
        i += 1
    for j in range(i, max_len):
        ret.append(0)
    return ret

# Actual Loaders #################################
def load_basic_data(fn):
	raw_data = pickle.load(open(fn, 'rb'))
	return raw_data


def build_data_set_from_year(year, months):
	full_data = [] 
	for m in months:
	    fn = "../data/pitches_{}_{}.p".format(year, m)
	    seqs = pickle.load(open(fn, "rb"))
	    full_data += seqs

	cleaned_data = [] 
	longest_seq = 0
	empties_or_single = 0
	pitch_types = set()
	for line in full_data:
	    if(len(line[1]) > longest_seq): longest_seq = len(line[1])
	    if(len(line[1]) <= 1): 
	        empties_or_single += 1
	    else:
	        cleaned_data.append(line)

	X_full = [] # Sequences of onehots.
	f_full = [] # Feature vectors
	y_full = [] # index of correct pitch in the one-hot, starting at X[1]
	for line in cleaned_data: 
		X_full.append(create_oneshot_seq(line, longest_seq))
		f_full.append(create_handpos_feature_vec(line))
		y_full.append(create_target(line, longest_seq))

	pickle.dump([X_full, f_full, y_full], open("../data/pitches_full_{}.p".format(year), "wb"))
	return X_full, f_full, y_full, longest_seq

