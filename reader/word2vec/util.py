import numpy as np
from vectorize import unitvec
from nltk.cluster.util import cosine_distance
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

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


def transform_word2vec_to_targets(predictions, vectors_by_codes, task):
    vector_by_code = {}
    for code, vectors in vectors_by_codes.items():
        prefix, code = code.split('_')
        if prefix == 'DRG' and task != 'drg': 
            continue
        if prefix == 'ICD' and task not in ['pdx', 'sdx']: 
            continue
        if prefix == 'CHOP' and task != 'srg': 
            continue
        data = np.zeros(vectors[0].shape[0], dtype=np.float32)
        # sum over all vectors (first vector is the code token)
        for t in vectors:
                data += t
        data = unitvec(data)
        vector_by_code[code] = data
    
    pred_targets = []
    for i in range(0, predictions.shape[0]):
        pred = predictions[i]
        target = ''
        min_cosine = float("inf")
        for code, vector in vector_by_code.items():
            cosine_distance = cosine(pred, vector)
            if cosine_distance < min_cosine:
                min_cosine = cosine_distance
                target = code
        pred_targets.append(target)

def accuracy(pred_targets, targets):
    num_true = 0.0
    for i, p in enumerate(pred_targets):
        print(p + ' => ' + targets[i])
        if(p == targets[i]):
            num_true += 1.0
    return num_true / len(pred_targets)


