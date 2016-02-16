from medcodelearn_pipeline import run
import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def validate_bool_var(bool_var, scores, options):
    config[bool_var] = False
    score = run(config)
    config[bool_var] = True
    score2 = run(config)
    scores.append(score2 - score)
    options.append(bool_var)


if __name__ == '__main__':
    base_folder = 'data/validate-options/'
    
    config = {
        'base_folder' : base_folder,
        # skip the word2vec vectorization step. Only possible if vectors have already been calculated.
        'skip-word2vec' : False,
        # classifier, one of 'random-forest', 'ffnn' (feed forward neural net) or 'lstm' (long short term memory, coming soon)
        'classifier' : 'ffnn',
        # Store all intermediate results. 
        # Disable this to speed up a run and to reduce disk space usage.
        'store-everything' : False,
        'drg-catalog' : 'data/2015/drgs.csv',
        'chop-catalog' : 'data/2015/chop_codes.csv',
        'icd-catalog' : 'data/2015/icd_codes.csv',
        'drg-tokenizations' : base_folder + 'tokenization/drgs_tokenized.csv',
        'icd-tokenizations' : base_folder + 'tokenization/icd_codes_tokenized.csv',
        'chop-tokenizations' : base_folder + 'tokenization/chop_codes_tokenized.csv',
        # Use the code descriptions for tokenization
        'use-descriptions' : True,
        'use-training-data-for-word2vec' : True,
        'shuffle-word2vec-traindata' : True,
        'num-shuffles' : 1,
        'all-tokens' : base_folder + 'tokenization/all_tokens.csv',
        'code-tokens' : base_folder + 'tokenization/all_tokens_by_code.json',
        'all-vocab' : base_folder + 'tokenization/vocab_all.csv',
        'all-vectors' : base_folder + 'vectorization/vectors.csv',
        'word2vec-dim-size' : 50,
        'word2vec-vocab': base_folder + 'vectorization/vocab.csv',
        'code-vectors' : base_folder + 'vectorization/all_vectors_by_code.json',
        'training-set-word2vec' : 'data/2015/trainingData2015_20151001.csv.last',
        'training-set' : 'data/2015/trainingData2015_20151001.csv.small',
        'training-set-drgs' : 'data/2015/trainingData2015_20151001.csv.out.small',
        # word2vec is deterministic only if non-parallelized. (Set num-cores to 1)
        'num-cores' : 8,
        # which demographic variables should be used.
        # a subset from ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 'ageDays']
        'demo-variables' : [] }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    scores = []
    options = []
    
    baseline = run(config)
        
    for bool_var in ['use-descriptions', 'use-training-data-for-word2vec', 'shuffle-word2vec-traindata']:
        validate_bool_var(bool_var, scores, options)
    
    temp = config['num-shuffles']
    config['num-shuffles'] = 10
    score = run(config)
    scores.append(score - baseline)
    options.append('num-shuffles=10')
    config['num-shuffles'] = 10
    
    config['skip-word2vec'] = True
    
    for demovar in ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 'ageDays']:
        config['demo-variables'] = [demovar]
        score = run(config)
        scores.append(score - baseline)
        options.append(demovar)
    
    print(options)    
    print(scores)
    
    y_pos = np.arange(len(options))
    
    plt.barh(y_pos, scores, align='center', alpha=0.4)
    plt.yticks(y_pos, options)
    plt.xlabel('Accuracy relative to Baseline')
    plt.title('Validation on different options')
    plt.grid(True)
    plt.savefig(base_folder + 'validate_options.pdf')