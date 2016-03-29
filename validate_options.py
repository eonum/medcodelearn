from medcodelearn_pipeline import run
import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def validate_bool_var(bool_var, scores, options, baseline):
    temp = config[bool_var]
    config[bool_var] = not temp
    score = run(config)
    config[bool_var] = temp
    diff = (baseline - score) if temp else (score - baseline)
    scores.append(diff)
    options.append(bool_var)
    visualize(scores, options)

def visualize(scores, options):
    print(options)    
    print(scores)
    
    y_pos = np.arange(len(options))
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.3)
    plt.barh(y_pos, scores, align='center', alpha=0.4)
    plt.yticks(y_pos, options)
    plt.xlabel('Accuracy relative to Baseline')
    plt.title('Validation on different options')
    plt.grid(True)
    plt.savefig(base_folder + 'validate_options.pdf')
    plt.close()

if __name__ == '__main__':
    base_folder = 'data/validate_activations/'
    
    config = {
        'base_folder' : base_folder,
        # skip the word2vec vectorization step. Only possible if vectors have already been calculated.
        'skip-word2vec' : True,
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
        'use_demographic_tokens' : True,
        # use skip grams (0) or CBOW (1) for word2vec
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
        'training-set-drgs' : 'data/2015/trainingData2015_20151001.csv.out.small',
        # word2vec is deterministic only if non-parallelized. (Set num-cores to 1)
        'num-cores' : 8,
        # which demographic variables should be used.
        # a subset from ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 'ageDays']
        'demo-variables' : ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 'ageDays','ageDays', 'adm-normal', 'adm-transfer',
                                               'adm-transfer-short',
                                               'sep-normal', 'sep-dead', 'sep-doctor',
                                               'sep-unknown', 'sep-transfer'],
	    'optimizer' : 'adam',
        'use-all-tokens-in-embedding' : False,
        # maximum sequence length for training
        'maxlen' : 32,
        'lstm-layers' : [{'output-size' : 64, 'dropout' : 0.1}],
        'outlayer-init' : 'glorot_uniform',
        'lstm-init' : 'glorot_uniform',
        'lstm-inner-init' : 'orthogonal',
        'lstm-activation' : 'sigmoid',
        'lstm-inner-activation' : 'hard_sigmoid' }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    json.dump(config, open(base_folder + 'configuration.json','w'), indent=4, sort_keys=True)    
    
    scores = []
    options = []
    
    baseline = run(config)
    
    inits = ['zero', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'uniform', 'lecun_uniform', 'normal', ]
    init_inner = ['identity', 'orthogonal']
    activations = ['linear', 'tanh', 'sigmoid', 'hard_sigmoid', 'relu', 'softplus']
    
    for activation in activations:
        config['lstm-activation'] = activation
        score = run(config)
        scores.append(score - baseline)
        options.append('lstm-activation-' + activation)
        visualize(scores, options)
        
    for bool_var in ['use-all-tokens-in-embedding', 'use-descriptions', 
                     'use-training-data-for-word2vec', 'shuffle-word2vec-traindata',
                     'word2vec-cbow', 'use_demographic_tokens', 'use-all-tokens-in-embedding']:
        validate_bool_var(bool_var, scores, options, baseline)
        
    
    
    temp = config['num-shuffles']
    config['num-shuffles'] = 1
    score1 = run(config)
    config['num-shuffles'] = 10
    score2 = run(config)
    scores.append(score2 - score1)
    options.append('num-shuffles=10')
    config['num-shuffles'] = temp
    
    visualize(scores, options)
    
    temp = config['word2vec-dim-size']
    config['word2vec-dim-size'] = 120
    score = run(config)
    config['num-shuffles'] = temp
    scores.append(score - baseline)
    options.append('word2vec-dim-size=120')
    
    visualize(scores, options)
    
    config['skip-word2vec'] = True

    for optimizer in ['adam', 'rmsprop']:
        config['optimizer'] = optimizer
        score = run(config)
        scores.append(score - baseline)
        options.append(optimizer)
        visualize(scores, options)


#     config['demo-variables'] = []
#     baseline_demo = run(config)
#     scores.append(baseline - baseline_demo)
#     options.append('all-demo-variables')
#     
#     for demovar in ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 
#                                                'ageDays', 'adm-normal', 'adm-transfer', 
#                                                'adm-transfer-short', 'adm-unknown',
#                                                'sep-normal', 'sep-dead', 'sep-doctor',
#                                                'sep-unknown', 'sep-transfer']:
#         config['demo-variables'] = [demovar]
#         score = run(config)
#         scores.append(score - baseline_demo)
#         options.append(demovar)
#         visualize(scores, options)

    
