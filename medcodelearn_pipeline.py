from tokenize_codes import tokenize_catalogs
import os
from subprocess import call
import json
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
    json.dump({k: v.tolist() for k, v in vector_by_token.items()}, open(config['all-vectors'] + '.json','w'), indent=4, sort_keys=True)
    # several vectors for each code. The first vector is from the code token.
    vectors_by_codes = read_code_vectors(vector_by_token, config['all-tokens'])
    json.dump(vectors_by_codes, open(config['code-vectors'],'w'), indent=4, sort_keys=True)
    
    
if __name__ == '__main__':
    base_folder = 'data/pipelinetest/'
    config = {
        'base_folder' : base_folder,
        'drg-catalog' : 'data/2015/drgs.csv',
        'chop-catalog' : 'data/2015/chop_codes.csv',
        'icd-catalog' : 'data/2015/icd_codes.csv',
        'drg-tokenizations' : base_folder + 'tokenization/drgs_tokenized.csv',
        'icd-tokenizations' : base_folder + 'tokenization/icd_codes_tokenized.csv',
        'chop-tokenizations' : base_folder + 'tokenization/chop_codes_tokenized.csv',
        'all-tokens' : base_folder + 'tokenization/all_tokens.csv',
        'all-vocab' : base_folder + 'tokenization/vocab_all.csv',
        'all-vectors' : base_folder + 'vectorization/vectors.csv',
        'word2vec-dim-size' : 50,
        'word2vec-vocab': base_folder + 'vectorization/vocab.csv',
        'code-vectors' : base_folder + 'vectorization/code-vectors.json' }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    
    run(config)
    