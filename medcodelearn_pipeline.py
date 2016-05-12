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
from classification.evaluation import adjust_score, plot_oracle, plot_classification_confidence_histograms
from reader.sequencevectors.pcreadersequencevectorized import SequenceVectorizedPCReader
from classification.lstm import train_and_evaluate_lstm, pad_sequences
from classification.lstmembedding import train_and_evaluate_lstm_with_embedding
from load_config import load_config
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
                                          use_n_times=config['num-shuffles'],
                                          use_demographic_tokens=config['use_demographic_tokens'])
            word2vec_trainset = base_folder + 'vectorization/train.txt'
        call(["word2vec", "-train", word2vec_trainset, "-binary",
               "0", "-cbow", '1' if config['word2vec-cbow'] else '0', "-output", config['all-vectors'],
                "-size", str(config['word2vec-dim-size']), "-save-vocab",
                config['word2vec-vocab'], "-min-count", "1", "-threads", str(config['num-cores'])])
    
    print("\nRead vectors. Assign vectors to codes..")
    # one vector for each token in the vocabulary
    vector_by_token = read_vectors(config['all-vectors'])
    vocab = vector_by_token.keys()
    
    if config['store-everything']:
        json.dump({k: v.tolist() for k, v in vector_by_token.items()}, open(config['all-vectors'] + '.json','w'), indent=4, sort_keys=True)

    res = read_code_vectors(vector_by_token, config['all-tokens'])
    
    # for each code a list of vectors of its tokens
    vectors_by_code = res['vectors']
    # for each code a list of its tokens
    tokens_by_code = res['tokens']
    # for each code a vector that is the normalized sum of all vectors from all tokens from this code.
    vector_by_code = res['vector_by_code']
    
    if config['store-everything']:
        json.dump({k: v.tolist() for k, v in vectors_by_code.items()}, open(config['code-vectors'],'w'), sort_keys=True)
        json.dump(tokens_by_code, open(config['code-tokens'],'w'), indent=4, sort_keys=True)
    
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
            reader.tokens_by_code = tokens_by_code
            reader.vocab = vocab
            reader.use_demographic_tokens = config['use_demographic_tokens']
            reader.use_all_tokens = config['use-all-tokens-in-embedding']
        else:
            reader = FlatVectorizedPCReader(config['training-set'])
        reader.read_from_file(vectors_by_code, task, drg_out_file=config['training-set-drgs'], demo_variables_to_use=config['demo-variables'])
        codes = reader.data
        targets = reader.targets
        excludes = reader.excludes
        demo_data = reader.demo_data
        classes = list(set(targets))
        y = np.zeros(len(codes), dtype=np.uint)
        for i, target in enumerate(targets):
            y[i] = classes.index(target)
        codes_train, codes_test, demo_train, demo_test, y_train, y_test, _, targets_test, _, excludes_test = train_test_split(codes, demo_data, y, targets, excludes, test_size=0.33, random_state=42)
        output_dim = len(set(targets))
        print('Number of classes: ' + str(output_dim))
        
        model, score = None, 0
        if config['classifier'] == 'random-forest':
            print("Training data dimensionality: " + str(codes.shape))
            print('Train Random Forest for ' + reader.code_type + ' classification task..')
            model, score = train_and_evaluate_random_forest(config, codes_train, codes_test, y_train, y_test)
        elif config['classifier'] == 'ffnn':
            print("Training data dimensionality: " + str(codes.shape))
            print('Train Feed Forward Neural Net for ' + reader.code_type + ' classification task..')
            model, scaler, score = train_and_evaluate_ffnn(config, codes_train, codes_test, y_train, y_test, output_dim, task)
            score = adjust_score(model, scaler, codes_test, classes, targets_test, excludes_test)
            plot_oracle(config, task, model, scaler, codes_test, classes, targets_test, excludes_test)
            plot_classification_confidence_histograms(config, task, model, scaler, codes_test, classes, targets_test, excludes_test)
        elif config['classifier'] == 'lstm':
            print("Training data dimensionality: " + str(len(codes)) + " | " + str(len(codes[0])) + " | " + str(len(codes[0][0])))
            print('Train LSTM Neural Net for ' + reader.code_type + ' classification task..')
            model, scaler, score = train_and_evaluate_lstm(config, codes_train, codes_test, y_train, y_test, output_dim, task)
            codes_test = pad_sequences(codes_test, maxlen=config['maxlen'], dim=len(codes_train[0][0]))
            score = adjust_score(model, scaler, codes_test, classes, targets_test, excludes_test)
        elif config['classifier'] == 'lstm-embedding':
            print("Training data dimensionality: " + str(len(codes)) + " | " + str(len(codes[0])))
            print('Train LSTM Neural Net with Embedding for ' + reader.code_type + ' classification task..')
            vocab = reader.vocab
            codes_train = keras.preprocessing.sequence.pad_sequences(codes_train, maxlen=config['maxlen'], dtype='int', truncating='pre')
            codes_test = keras.preprocessing.sequence.pad_sequences(codes_test, maxlen=config['maxlen'], dtype='int', truncating='pre')
                 
            model, score = train_and_evaluate_lstm_with_embedding(config, codes_train, codes_test, demo_train, demo_test, y_train, y_test, output_dim, task, vocab, 
                                                                  vector_by_token,
                                                                  vector_by_code)
            input_test = {'codes_input':codes_test, 'demo_input':demo_test}
            score = adjust_score(model, None, input_test, classes, targets_test, excludes_test)
            plot_oracle(config, task, model, None, input_test, classes, targets_test, excludes_test)
            plot_classification_confidence_histograms(config, task, model, None, input_test, classes, targets_test, excludes_test)

        total_score += score
        if config['store-everything']:
            joblib.dump(model, base_folder + 'classification/' + config['classifier'] + '.pkl')
    
    total_score /= len(tasks)
    print('Total average score over all tasks: ' + str(total_score))
    return total_score
    
    
if __name__ == '__main__':
    config = load_config()
    base_folder = config['base_folder']
    
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    run(config)
    
