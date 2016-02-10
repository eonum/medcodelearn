from medcodelearn_pipeline import run
import json
import os
import matplotlib.pyplot as plt
from bokeh.io import show

if __name__ == '__main__':
    base_folder = 'data/validate-word2vec-dimsize/'
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
        'training-set' : 'data/2015/trainingData2015_20151001.csv.small',
        'training-set-drgs' : 'data/2015/trainingData2015_20151001.csv.out.small',
        'num-cores' : 4 }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    scores = []
    params = []
    for i in range(1, 200):
        print("Validate parameter " + str(i))
        config['word2vec-dim-size'] = i
        score = run(config)
        scores.append(score)
        params.append(i)
    
    print(params)    
    print(scores)
    
    plt.plot(params, scores)
    plt.title('validation on the size of the word2vec dimensionality')
    plt.ylabel('accuracy')
    plt.xlabel('vector size')
    plt.grid(True)
    plt.savefig(base_folder + 'word2vec_dimensions_validation.pdf')
