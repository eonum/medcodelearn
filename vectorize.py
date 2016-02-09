# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA

def read_vectors(fname, vocabUnicodeSize=78, desired_vocab=None, encoding="utf-8"):
    """
    Create a vectors for each token based on a word2vec text file
    Parameters
    ----------
    fname : path to file
    vocabUnicodeSize: the maximum string length (78, by default)
    desired_vocab: if set, this will ignore any word and vector that
                   doesn't fall inside desired_vocab.
    Returns
    -------
    dict vector by token
    """
    with open(fname, 'rb') as fin:
        vectors = {}
        for line in fin:
            line = line.decode(encoding).strip()
            parts = line.split(' ')
            word = parts[0]
            include = desired_vocab is None or word in desired_vocab
            if include:
                vector = np.array(parts[1:], dtype=np.float)
                vectors[word] = unitvec(vector)
    return vectors
    
def unitvec(vec):
    norm = LA.norm(vec, ord=2)
    return vec if norm == 0 else (1.0 / norm) * vec

def read_code_vectors(vector_by_token, code_token_file, encoding="utf-8"): 
    with open(code_token_file, 'rb') as fin:
        vectors = {}
        tokens = {}
        for line in fin:
            line = line.decode(encoding).strip()
            ts = line.split(' ')
            tokens[ts[0]] = ts
            vs =  np.empty((len(ts), len(vector_by_token[ts[0]])), dtype=np.float)
            for i, token in enumerate(ts):
                # empty token
                token = '</s>' if token == '' else token
                vs[i] = vector_by_token[token]
            vectors[ts[0]] = vs
    return {'vectors' : vectors, 'tokens' : tokens} 
