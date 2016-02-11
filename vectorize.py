# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
from reader.sparsehierarchical.drgreader import DRGReader
import csv
import random

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


def create_word2vec_training_data(train_file, token_by_code_file, out_file_name, encoding="utf-8", do_shuffle=False, use_n_times=1):
    tokens_by_code = {}
    out_file = open(out_file_name, 'w')

    with open(token_by_code_file, 'rb') as fin:
        for line in fin:
            line = line.decode(encoding).strip()
            ts = line.split(' ')
            tokens_by_code[ts[0]] = ts
            # Use each code at least once
            for t in ts:
                out_file.write(t + ' ')
            out_file.write("\n")
    
    with open(train_file, 'r') as csvFile:
        reader = csv.DictReader(csvFile, fieldnames=DRGReader.FIELDNAMES, restkey=DRGReader.RESTKEY, delimiter=';')
        for row in reader:
            diagproc = row[DRGReader.RESTKEY]
            diags = [row['pdx']] + diagproc[0:DRGReader.MAX_ADDITIONAL_DIAGNOSES]
            procs = map(lambda x: x.split(':')[0], diagproc[DRGReader.MAX_ADDITIONAL_DIAGNOSES:DRGReader.MAX_ADDITIONAL_DIAGNOSES+DRGReader.MAX_PROCEDURES])
            procs = list(map(lambda x: 'CHOP_' + x.replace('.', '').upper(), procs))
            diags = list(map(lambda x: 'ICD_' + x.replace('.', '').upper(), diags))
            diagproc = diags + procs
            diagproc = [p for p in diagproc if p in tokens_by_code]
            
            for i in range(use_n_times):
                random.seed(i)
                r = random.random()
                if do_shuffle:
                    random.shuffle(diagproc, lambda: r)
                for d in diagproc:
                    for t in tokens_by_code[d]:
                        out_file.write(t + ' ')         
                out_file.write("\n")

    out_file.close()
