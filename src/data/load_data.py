import pickle 
import numpy as np
import random 

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
def create_handposbase_feature_vec(seq):
    # Create positions one-hot
    pos = np.zeros((len(POS_MAP),), dtype=np.float32)
    for p in seq[0][2].split("-"):
        pos[POS_MAP[p]] = 1.0
    
    # Handedness features
    hands = np.zeros((2,), dtype=np.float32)
    hands[0] = HAND_MAP[seq[0][0]]
    hands[1] = HAND_MAP[seq[0][1]]

    # on base 
    base = seq[0][3]
    return np.concatenate((hands, pos, base))
    
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

def create_window_x(seq, ws):
    features = np.array(create_handposbase_feature_vec(seq)).flatten()
    seqs = []

    for i in range(len(seq[1])):
        sub_seq = []
        if i < ws-1:
            for zeros in range(ws-i):
                sub_seq.append(np.zeros((len(PITCH_MAP),), dtype=np.float32))
            start = ws-i
        else:
            start = i-ws

        for j in range(start, i):
            p_oh = np.zeros((len(PITCH_MAP),), dtype=np.float32)
            p = seq[1][j]
            p_oh[PITCH_MAP[p]] = 1.0
            sub_seq.append(p_oh)
        sub_seq = np.array(sub_seq).flatten()
        seqs.append(np.concatenate((features, sub_seq)))
    return seqs

def create_window_y(seq, ws):
    ys = []
    for i in range(len(seq[1])-1):
        p_oh = np.zeros((len(PITCH_MAP),), dtype=np.float32)
        p = seq[1][i+1]
        p_oh[PITCH_MAP[p]] = 1.0
        ys.append(p_oh)
    ys.append(np.zeros((len(PITCH_MAP),), dtype=np.float32))
    return ys

def build_data_set_from_years(years, months):
    full_data = [] 
    for y in years:
        for m in months:
            fn = "../data/pitches_{}_{}.p".format(y, m)
            try:
                seqs = pickle.load(open(fn, "rb"))
                full_data += seqs
            except:
                print("error opening {}".format(fn))

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
    errs = 0
    for line in cleaned_data: 
        try:
            X_full.append(create_oneshot_seq(line, longest_seq))
            f_full.append(create_handposbase_feature_vec(line))
            y_full.append(create_target(line, longest_seq))
        except:
            errs += 1
    print("{} lines had errors".format(errs))

    return X_full, f_full, y_full, longest_seq

def build_fake_sequences(length, alpha, period, reps):
    ret = []
    for i in range(length):
        seq = ""
        for p in range(period):
            letter = random.choice(alpha)
            seq += letter*reps
        ret.append(seq)
    return ret

def build_fake_data_set(seqs, alpha_map, max_len):
    X = []
    f = []
    y = []
    oh_len = len(seqs[0])
    for seq in seqs:
        ex = []
        target = []
        for c in seq:
            s_oh = np.zeros((len(alpha_map),), dtype=np.float32)
            s_oh[alpha_map[c]] = 1.0
            ex.append(s_oh)
        X.append(ex)
        f.append([0])
    
    for seq in seqs:
        target = []
        for c in seq[1:]:
            target.append(alpha_map[c])
        target.append(0)
        y.append(target)
    return X, f, y

def build_window_data_set(years, months, window_size):
    full_data = [] 
    for y in years:
        for m in months:
            fn = "../../data/partials/pitches_{}_{}.p".format(y, m)
            try:
                seqs = pickle.load(open(fn, "rb"))
                full_data += seqs
            except:
                print("error opening {}".format(fn))

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

    X = [] # an N by 2*window_size+19 array (2(handedness)+14(pos)+3(base))
    y = [] # a N by num_pitches output (one_hot of the pitch)

    for line in cleaned_data:
        try:
            X += create_window_x(line, window_size)
            y += create_window_y(line, window_size)
        except:
            print("error with line: {}".format(line))
    return X, y