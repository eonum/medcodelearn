# -*- coding: utf-8 -*-
import word2vec

if __name__  == "__main__":
    word2vec.word2vec('data/chop_codes_tokenized.txt', 'data/chop_codes.bin', size=150, verbose=True)
    word2vec.word2vec('data/drgs_tokenized.txt', 'data/drgs.bin', size=150, verbose=True)
    word2vec.word2vec('data/icd_codes_tokenized.txt', 'data/icd_codes.bin', size=150, verbose=True)    