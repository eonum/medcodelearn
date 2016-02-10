from tokenization.tokenize_codes import tokenize_catalogs
import os
from subprocess import call
import json
from json import encoder
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from reader.flatvectors.drgreaderflatvectorized import FlatVectorizedDRGReader
from classification.random_forest import train_and_evaluate_random_forest
encoder.FLOAT_REPR = lambda o: format(o, '.8f')

from vectorize import read_code_vectors, read_vectors


def run (config):
    base_folder = config['base_folder']
    
    print("Tokenize catalogs..")
    if not os.path.exists(base_folder + 'tokenization'):
        os.makedirs(base_folder + 'tokenization')
    tokenize_catalogs(config)
    
    print("Vectorize catalogs..")
    if not os.path.exists(base_folder + 'vectorization'):
        os.makedirs(base_folder + 'vectorization')
    call(["word2vec", "-train", config['all-tokens'], "-binary",
           "0", "-cbow", "0", "-output", config['all-vectors'],
            "-size", str(config['word2vec-dim-size']), "-save-vocab",
            config['word2vec-vocab'], "-min-count", "1"])
    
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
        
    print('Read patient cases..')
    reader = FlatVectorizedDRGReader(config['training-set'])
    reader.read_from_file(vectors_by_codes, 'sdx')
    data = reader.data
    targets = reader.targets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.33, random_state=42)
    print("Training data dimensionality: " + str(data.shape))
    
    print('Train Random Forest for DRG Code Proposals..')
    rf_model, score = train_and_evaluate_random_forest(config, X_train, X_test, y_train, y_test)
    if config['store-everything']:
        if not os.path.exists(base_folder + 'classification'):
            os.makedirs(base_folder + 'classification')
        joblib.dump(rf_model, base_folder + 'classification/random-forest.pkl')
    
    
    return score
    
    
if __name__ == '__main__':
    base_folder = 'data/pipelinetest/'
    config = {
        'base_folder' : base_folder,
        # Store all intermediate results. 
        # Disable this to speed up a run and to reduce disk space usage.
        'store-everything' : False,
        'drg-catalog' : 'data/2015/drgs.csv',
        'chop-catalog' : 'data/2015/chop_codes.csv',
        'icd-catalog' : 'data/2015/icd_codes.csv',
        'drg-tokenizations' : base_folder + 'tokenization/drgs_tokenized.csv',
        'icd-tokenizations' : base_folder + 'tokenization/icd_codes_tokenized.csv',
        'chop-tokenizations' : base_folder + 'tokenization/chop_codes_tokenized.csv',
        'all-tokens' : base_folder + 'tokenization/all_tokens.csv',
        'code-tokens' : base_folder + 'tokenization/all_tokens_by_code.json',
        'all-vocab' : base_folder + 'tokenization/vocab_all.csv',
        'all-vectors' : base_folder + 'vectorization/vectors.csv',
        'word2vec-dim-size' : 50,
        'word2vec-vocab': base_folder + 'vectorization/vocab.csv',
        'code-vectors' : base_folder + 'vectorization/all_vectors_by_code.json',
        'training-set' : 'data/2015/trainingData2015_20151001.csv',
        'training-set-drgs' : 'data/2015/trainingData2015_20151001.csv.out.small',
        'num-cores' : 4 }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    
    run(config)
    