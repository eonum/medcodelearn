import numpy as np
from vectorize import unitvec

def transform_targets_to_word2vec(targets, task, vectors_by_codes, word2vec_size):
    y = np.empty((len(targets), word2vec_size), dtype=np.float32)
    for i, target in enumerate(targets):
        prefix = 'ICD_' if task in ['pdx', 'sdx'] else 'CHOP_' if task == 'srg' else 'DRG_'
        vectors = vectors_by_codes[prefix + target]
        data = np.zeros(word2vec_size, dtype=np.float32)
        # sum over all vectors (first vector is the code token)
        for t in vectors:
                data += t
        data = unitvec(data)
        y[i] = data
    return y


