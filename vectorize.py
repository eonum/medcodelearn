# -*- coding: utf-8 -*-
import word2vec
import os
# mysteriously doesn't work anymore... instead using vectorize.sh at the moment.
if __name__  == "__main__":
    datapath = os.path.join(os.path.dirname(__file__), 'data')
    word2vec.word2vec(os.path.join(datapath,'tokenization', 'tokens.csv'), os.path.join(datapath,'vectorization', 'tokens.bin'), size=50, verbose=True)   