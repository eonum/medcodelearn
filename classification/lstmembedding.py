import numpy as np
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from keras.models import Graph
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout, Dense
import json
from keras.callbacks import EarlyStopping
from classification.LossHistoryVisualization import LossHistoryVisualisation
from keras.layers.embeddings import Embedding
from classification.GraphMonitor import GraphMonitor


def train_and_evaluate_lstm_with_embedding(config, X_train, X_test, y_train, y_test, output_dim, task, vocab, vector_by_token):
    y_train = np_utils.to_categorical(y_train, output_dim)
    y_test = np_utils.to_categorical(y_test, output_dim)
    
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=23)
    
    n_symbols = len(vocab)
    embedding_weights = np.zeros((n_symbols, config['word2vec-dim-size']), dtype=np.float32)
    for index, word in enumerate(vocab):
        # skip first item 'mask'
        if index == 0 or word == '':
            continue
        embedding_weights[index,:] = vector_by_token[word]
       
    model = Graph()
    model.add_input(name='codes_input', input_shape=(config['maxlen'],), dtype='int')
    model.add_node(Embedding(n_symbols, config['word2vec-dim-size'], input_length=config['maxlen'], 
                             mask_zero=True, weights=[embedding_weights]), 
                             name='embedding', input='codes_input')
    node = 'embedding'
    for i, layer in enumerate(config['lstm-layers']):
        inputnode = node
        node = 'lstm' + str(i)
        model.add_node(LSTM(output_dim=layer['output-size'], activation='sigmoid', 
                            inner_activation='hard_sigmoid',
                            return_sequences=i != len(config['lstm-layers']) - 1),
                            name=node, input=inputnode)
        inputnode = node
        node = 'dropout' + str(i)
        model.add_node(Dropout(layer['dropout']), name=node, input=inputnode)
    
    model.add_node(Dense(output_dim, activation='softmax'), name='softmax', input=node)
    model.add_output(name='output', input='softmax')
    
    model.compile(loss={'output' : 'categorical_crossentropy'},
                  optimizer=config['optimizer'])
    
    json.dump(json.loads(model.to_json()), 
              open(config['base_folder'] + 'classification/model_lstm_' + task + '.json','w'), indent=4, sort_keys=True)   

    
    graphmonitor = GraphMonitor(config['base_folder'] + 'classification/', task_name=task, patience=10, output_names=['output'])
    model.fit({'codes_input':X_train, 'output':y_train},
              nb_epoch=50,
              batch_size=128,
              validation_data={'codes_input':X_validation, 'output':y_validation},
              verbose=2,
              callbacks=[graphmonitor])
    
    print("Prediction using LSTM..")
    score = model.evaluate({'codes_input':X_test, 'output':y_test}, verbose=0)
    print('Test score:', score)

    return [model, score]
