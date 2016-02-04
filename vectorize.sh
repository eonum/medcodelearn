# mkdir data/vectorization if not already exists
word2vec -train data/tokenization/tokens.csv -binary 0 -cbow 0 -output data/vectorization/vectors.csv -size 50 -save-vocab data/vectorization/vocab.csv -min-count 1
