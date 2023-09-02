from sequence_hopfield_utils import ContinuousHopfield
import string
import numpy as np

all_chars = string.ascii_letters + string.digits + string.punctuation + ' []'


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def convert_string_pattern(string_pattern, decay_rate=0.9):
    pattern_vecs = []
    for char in string_pattern:
        pattern_vec = [0]*len(all_chars)
        if len(pattern_vecs) > 0:
            pattern_vec = [p*decay_rate for p in pattern_vecs[-1]]
        ind = all_chars.index(char)
        pattern_vec[ind] += 1
        pattern_vecs.append(normalize(pattern_vec))
    next_pattern_vecs = pattern_vecs[1:] + [[1]*len(all_chars)]
    pattern_vecs = [np.array(p).reshape(-1, 1) for p in pattern_vecs]
    next_pattern_vecs = [np.array(p).reshape(-1, 1) for p in next_pattern_vecs]
    return np.array(pattern_vecs), np.array(next_pattern_vecs)
