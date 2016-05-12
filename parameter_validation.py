from medcodelearn_pipeline import run
import json
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from load_config import load_config

def plot_scores(params, scores):
    plt.plot(params, scores)
    plt.title('validation on the size of the word2vec dimensionality')
    plt.ylabel('accuracy')
    plt.xlabel('vector size')
    plt.grid(True)
    plt.savefig(base_folder + 'word2vec_dimensions_validation.pdf')

if __name__ == '__main__':
    config = load_config()
    base_folder = config['base_folder']

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
  
