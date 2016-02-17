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
    codes = []
    prefix = 'DRG' if task == 'drg' else 'CHOP' if task == 'srg' else 'ICD'
    for code in vectors_by_codes:
        p, code = code.split('_')
        if p != prefix: 
            continue
        
        codes.append(code)
    
    vectors = np.empty((len(codes), 50), dtype=np.float32)    
    for i, code in enumerate(codes):
        vs = vectors_by_codes[prefix + '_' + code]
        data = np.zeros(vs[0].shape[0], dtype=np.float32)
        # sum over all vectors (first vector is the code token)
        for t in vs:
                data += t
        data = unitvec(data)
        vectors[i] = data
    
    pred_targets = []
    for i in range(0, predictions.shape[0]):
        pred = predictions[i]
        metrics = np.dot(vectors, pred)
        best = np.argsort(metrics)[::1][0:4]  
        target = codes[best[0]]      
        pred_targets.append(target)
        # for probabilities
        # best_metrics = metrics[best]
            
    return pred_targets

def accuracy(pred_targets, targets):
    num_true = 0.0
    for i, p in enumerate(pred_targets):
        if(p == targets[i]):
            num_true += 1.0
    return num_true / len(pred_targets)


