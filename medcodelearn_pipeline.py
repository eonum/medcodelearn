from tokenization.tokenize_codes import tokenize_catalogs
import os
from subprocess import call
import json
from json import encoder
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import numpy as np

from reader.flatvectors.pcreaderflatvectorized import FlatVectorizedPCReader
from classification.random_forest import train_and_evaluate_random_forest
from classification.ffnn import train_and_evaluate_ffnn
from classification.ffnn import adjust_score
from reader.sequencevectors.pcreadersequencevectorized import SequenceVectorizedPCReader
from classification.lstm import train_and_evaluate_lstm, pad_sequences
from classification.lstmembedding import train_and_evaluate_lstm_with_embedding
from reader.sequence.pcreadersequence import SequencePCReader
import keras.preprocessing.sequence

encoder.FLOAT_REPR = lambda o: format(o, '.8f')

from vectorize import read_code_vectors, read_vectors, create_word2vec_training_data


def run (config):
    base_folder = config['base_folder']
    
    if not config['skip-word2vec']:
        print("Tokenize catalogs..")
        if not os.path.exists(base_folder + 'tokenization'):
            os.makedirs(base_folder + 'tokenization')
        tokenize_catalogs(config)
        
        print("Vectorize catalogs..")
        if not os.path.exists(base_folder + 'vectorization'):
            os.makedirs(base_folder + 'vectorization')
        word2vec_trainset = config['all-tokens']
        if config['use-training-data-for-word2vec']:
            create_word2vec_training_data(config['training-set-word2vec'], config['all-tokens'], 
                                          base_folder + 'vectorization/train.txt',
                                          do_shuffle=config['shuffle-word2vec-traindata'],
                                          use_n_times=config['num-shuffles'])
            word2vec_trainset = base_folder + 'vectorization/train.txt'
        call(["word2vec", "-train", word2vec_trainset, "-binary",
               "0", "-cbow", '1' if config['word2vec-cbow'] else '0', "-output", config['all-vectors'],
                "-size", str(config['word2vec-dim-size']), "-save-vocab",
                config['word2vec-vocab'], "-min-count", "1", "-threads", str(config['num-cores'])])
    
    print("\nRead vectors. Assign vectors to codes..")
    # one vector for each token in the vocabulary
    vector_by_token = read_vectors(config['all-vectors'])
    if config['store-everything']:
        json.dump({k: v.tolist() for k, v in vector_by_token.items()}, open(config['all-vectors'] + '.json','w'), indent=4, sort_keys=True)
    # several vectors for each code. The first vector is from the code token.
    res = read_code_vectors(vector_by_token, config['all-tokens'])
    vectors_by_codes = res['vectors']
    tokens_by_codes = res['tokens']
    if config['store-everything']:
        json.dump({k: v.tolist() for k, v in vectors_by_codes.items()}, open(config['code-vectors'],'w'), sort_keys=True)
        json.dump(tokens_by_codes, open(config['code-tokens'],'w'), indent=4, sort_keys=True)
    
    if not os.path.exists(base_folder + 'classification'):
        os.makedirs(base_folder + 'classification')
    total_score = 0.0 
    tasks = ['pdx', 'sdx', 'srg', 'drg']   
    for task in tasks:
        print('\n==== ' + task + ' ====')
        reader = None
        if config['classifier'] == 'lstm':
            reader = SequenceVectorizedPCReader(config['training-set'])
        elif config['classifier'] == 'lstm-embedding':
            reader = SequencePCReader(config['training-set'])
            reader.tokens_by_code = tokens_by_codes
        else:
            reader = FlatVectorizedPCReader(config['training-set'])
        reader.read_from_file(vectors_by_codes, task, drg_out_file=config['training-set-drgs'], demo_variables_to_use=config['demo-variables'])
        X = reader.data
        targets = reader.targets
        excludes = reader.excludes
        classes = list(set(targets))
        y = np.zeros(len(X), dtype=np.uint)
        for i, target in enumerate(targets):
            y[i] = classes.index(target)
        X_train, X_test, y_train, y_test, _, targets_test, _, excludes_test = train_test_split(X, y, targets, excludes, test_size=0.33, random_state=42)
        output_dim = len(set(targets))
        print('Number of classes: ' + str(output_dim))
        
        model, score = None, 0
        if config['classifier'] == 'random-forest':
            print("Training data dimensionality: " + str(X.shape))
            print('Train Random Forest for ' + reader.code_type + ' classification task..')
            model, score = train_and_evaluate_random_forest(config, X_train, X_test, y_train, y_test)
        elif config['classifier'] == 'ffnn':
            print("Training data dimensionality: " + str(X.shape))
            print('Train Feed Forward Neural Net for ' + reader.code_type + ' classification task..')
            model, scaler, score = train_and_evaluate_ffnn(config, X_train, X_test, y_train, y_test, output_dim, task)
            score = adjust_score(model, scaler, X_test, classes, targets_test, excludes_test)
        elif config['classifier'] == 'lstm':
            print("Training data dimensionality: " + str(len(X)) + " | " + str(len(X[0])) + " | " + str(len(X[0][0])))
            print('Train LSTM Neural Net for ' + reader.code_type + ' classification task..')
            model, scaler, score = train_and_evaluate_lstm(config, X_train, X_test, y_train, y_test, output_dim, task)
            X_test = pad_sequences(X_test, maxlen=17, dim=len(X_train[0][0]))
            score = adjust_score(model, scaler, X_test, classes, targets_test, excludes_test)
        elif config['classifier'] == 'lstm-embedding':
            print("Training data dimensionality: " + str(len(X)) + " | " + str(len(X[0])))
            print('Train LSTM Neural Net with Embedding for ' + reader.code_type + ' classification task..')
            vocab = reader.vocab
            X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=150, dtype='int32', truncating='post')
            X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=150, dtype='int32', truncating='post')
                 
            model, score = train_and_evaluate_lstm_with_embedding(config, X_train, X_test, y_train, y_test, output_dim, task, vocab, vector_by_token)
            score = adjust_score(model, None, X_test, classes, targets_test, excludes_test)
        
        total_score += score
        if config['store-everything']:
            joblib.dump(model, base_folder + 'classification/' + config['classifier'] + '.pkl')
    
    total_score /= len(tasks)
    print('Total average score over all tasks: ' + str(total_score))
    return total_score
    
    
if __name__ == '__main__':
    base_folder = 'data/pipelinetest/'
    config = {
        'base_folder' : base_folder,
        # skip the word2vec vectorization step. Only possible if vectors have already been calculated.
        'skip-word2vec' : False,
        # classifier, one of 'random-forest', 'ffnn' (feed forward neural net) or 'lstm' (long short term memory, coming soon)
        'classifier' : 'lstm-embedding',
        # Store all intermediate results. 
        # Disable this to speed up a run and to reduce disk space usage.
        'store-everything' : False,
        'drg-catalog' : 'data/2015/drgs.csv',
        'chop-catalog' : 'data/2015/chop_codes.csv',
        'icd-catalog' : 'data/2015/icd_codes.csv',
        'drg-tokenizations' : base_folder + 'tokenization/drgs_tokenized.csv',
        'icd-tokenizations' : base_folder + 'tokenization/icd_codes_tokenized.csv',
        'chop-tokenizations' : base_folder + 'tokenization/chop_codes_tokenized.csv',
        # use skip grams (False) or CBOW (True) for word2vec
        'word2vec-cbow' : True,
        # Use the code descriptions for tokenization
        'use-descriptions' : False,
        'use-training-data-for-word2vec' : True,
        'shuffle-word2vec-traindata' : True,
        'num-shuffles' : 10,
        'all-tokens' : base_folder + 'tokenization/all_tokens.csv',
        'code-tokens' : base_folder + 'tokenization/all_tokens_by_code.json',
        'all-vocab' : base_folder + 'tokenization/vocab_all.csv',
        'all-vectors' : base_folder + 'vectorization/vectors.csv',
        'word2vec-dim-size' : 50,
        'word2vec-vocab': base_folder + 'vectorization/vocab.csv',
        'code-vectors' : base_folder + 'vectorization/all_vectors_by_code.json',
        'training-set-word2vec' : 'data/2015/trainingData2015_20151001.csv.last',
        'training-set' : 'data/2015/trainingData2015_20151001.csv.small',
        'training-set-drgs' : 'data/2015/trainingData2015_20151001.csv.small.out',
        # word2vec is deterministic only if non-parallelized. (Set num-cores to 1)
        'num-cores' : 8,
        # which demographic variables should be used.
        # a subset from ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 'ageDays']
        'demo-variables' : ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 
                                               'ageDays', 'adm-normal', 'adm-transfer', 
                                               'adm-transfer-short', 'adm-unknown',
                                               'sep-normal', 'sep-dead', 'sep-doctor',
                                               'sep-unknown', 'sep-transfer'],
        # NN optimizer, one of ['sgd', 'rmsprop', 'adam', 'adagrad', 'adadelta', 'adamax']
        'optimizer' : 'adam' }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    
    run(config)
    
