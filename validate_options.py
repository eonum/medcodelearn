from medcodelearn_pipeline import run
import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from load_config import load_config


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
    config = load_config()
    base_folder = config['base_folder']

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

    
