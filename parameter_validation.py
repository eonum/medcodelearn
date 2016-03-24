from medcodelearn_pipeline import run
import json
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_scores(params, scores):
    plt.plot(params, scores)
    plt.title('validation on the size of the word2vec dimensionality')
    plt.ylabel('accuracy')
    plt.xlabel('vector size')
    plt.grid(True)
    plt.savefig(base_folder + 'word2vec_dimensions_validation.pdf')

if __name__ == '__main__':
    base_folder = 'data/lstm-hiddennodesize/'
    config = {
        'base_folder' : base_folder,
        # skip the word2vec vectorization step. Only possible if vectors have already been calculated.
        'skip-word2vec' : True,
        # classifier, one of 'random-forest', 'ffnn' (feed forward neural net), 'lstm', 'lstm-embedding'
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
        'use_demographic_tokens' : True,
        # use skip grams (False) or CBOW (True) for word2vec
        'word2vec-cbow' : True,
        # Use the code descriptions for tokenization
        'use-descriptions' : True,
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
        'optimizer' : 'adam',
        # Whether to use all tokens for the LSTM embedding or only codes (normalized sum over all vectors)
        'use-all-tokens-in-embedding' : False,
        # maximum sequence length for training
        'maxlen' : 32,
        'lstm-layers' : [{'output-size' : 64, 'dropout' : 0.1}] }

    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    scores = []
    params = []
    for i in range(1, 250, 10):
        print("Validate parameter " + str(i))
        config['lstm-layers'][0]['output-size'] = i
        score = run(config)
        scores.append(score)
        params.append(i)
        plot_scores(params, scores)
    
    print(params)    
    print(scores)
    
    plot_scores(params, scores)
  
