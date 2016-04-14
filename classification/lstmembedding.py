import numpy as np
import json
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.layers import Dropout, Dense, Input, LSTM, Embedding
from keras.models import Model

from classification.GraphMonitor import GraphMonitor

from keras.callbacks import EarlyStopping
from classification.LossHistoryVisualization import LossHistoryVisualisation


def train_and_evaluate_lstm_with_embedding(config, X_train, X_test, y_train, y_test, output_dim, task, vocab, vector_by_token, vector_by_code):
    y_train = np_utils.to_categorical(y_train, output_dim)
    y_test = np_utils.to_categorical(y_test, output_dim)
    
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=23)
    
    n_symbols = len(vocab)
    embedding_weights = np.zeros((n_symbols, config['word2vec-dim-size']), dtype=np.float32)
    for index, word in enumerate(vocab):
        # skip first item 'mask'
        if index == 0 or word == '':
            continue
        embedding_weights[index,:] = vector_by_token[word] if config['use-all-tokens-in-embedding'] or word not in vector_by_code else vector_by_code[word]
       

    codes_input = Input(shape=(config['maxlen'],), dtype='int32', name='codes_input')
    embedding = Embedding(n_symbols, config['word2vec-dim-size'], input_length=config['maxlen'], 
                             mask_zero=True, weights=[embedding_weights])(codes_input)
    node = embedding
    for i, layer in enumerate(config['lstm-layers']):
        node = LSTM(output_dim=layer['output-size'], activation='sigmoid', 
                            inner_activation='hard_sigmoid',
                            return_sequences=i != len(config['lstm-layers']) - 1)(node)
        node = Dropout(layer['dropout'])(node)
    
    output = Dense(output_dim, activation='softmax', name='output')(node)
    
    model = Model(input=[codes_input], output=[output])
    
    model.compile(loss={'output' : 'categorical_crossentropy'},
                  optimizer=config['optimizer'],
                  metrics=['accuracy'])
    
    json.dump(json.loads(model.to_json()), 
              open(config['base_folder'] + 'classification/model_lstm_' + task + '.json','w'), indent=4, sort_keys=True)   

    
    early_stopping = EarlyStopping(monitor='val_acc', patience=10)
    visualizer = LossHistoryVisualisation(config['base_folder'] + 'classification/epochs_' + task + '.png')
    model.fit({'codes_input':X_train}, {'output':y_train},
              nb_epoch=50,
              batch_size=128,
              validation_data=({'codes_input':X_validation}, {'output':y_validation}),
              verbose=2,
              callbacks=[early_stopping, visualizer])
    
    print("Prediction using LSTM..")
    score = model.evaluate({'codes_input':X_test}, {'output':y_test}, verbose=0)
    print('Test score:', score)

    return [model, score]
