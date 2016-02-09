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
            header = fin.readline()

            vectors = {}
            for i, line in enumerate(fin):
                line = line.decode(encoding).strip()
                parts = line.split(' ')
                word = parts[0]
                include = desired_vocab is None or word in desired_vocab
                if include:
                    vector = np.array(parts[1:], dtype=np.float)
                    vectors[word] = unitvec(vector)

        return vectors
    
def unitvec(vec):
    return (1.0 / LA.norm(vec, ord=2)) * vec

def read_code_vectors(vector_by_token, code_token_file): 
    return{}  
