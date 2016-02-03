# mkdir data/vectorization if not already exists
/home/tim/word2vec/word2vec -train data/tokenization/tokens.csv -binary 1 -cbow 0 -output data/vectorization/vectors.bin -size 50 -save-vocab data/vectorization/vocab.csv -min-count 1
