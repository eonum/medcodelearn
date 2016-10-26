import numpy as np
import json
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.layers import Dropout, Dense, Input, LSTM, Embedding, merge
from keras.models import Model

from keras.callbacks import EarlyStopping
from classification.LossHistoryVisualization import LossHistoryVisualisation


def train_and_evaluate_lstm_with_embedding(config, codes_train, codes_test, demo_train, demo_test, y_train, y_test, output_dim, task, vocab, vector_by_token, vector_by_code):
    if task != 'los':
        y_train = np_utils.to_categorical(y_train, output_dim)
        y_test = np_utils.to_categorical(y_test, output_dim)
    
    
    codes_train, codes_validation, demo_train, demo_validation, y_train, y_validation = train_test_split(codes_train, demo_train, y_train, test_size=0.15, random_state=23)
    
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
        node = LSTM(output_dim=layer['output-size'], activation=config['lstm-activation'], 
                            inner_activation=config['lstm-inner-activation'], init=config['lstm-init'],
                            inner_init=config['lstm-inner-init'],
                            return_sequences=i != len(config['lstm-layers']) - 1)(node)
        node = Dropout(layer['dropout'])(node)
    
    demo_input = Input(shape=(len(config['demo-variables']),), name='demo_input')
    node = merge([node, demo_input], mode='concat')
    node = Dense(64, activation='relu')(node)

    if task == 'los':
        output = Dense(1, init=config['outlayer-init'], name='output')(node)
    else:
        output = Dense(output_dim, activation='softmax', init=config['outlayer-init'], name='output')(node)
        
    
    model = Model(input=[codes_input, demo_input], output=[output])
    
    if task == 'los':
        additional_metric_name = 'mean_absolute_percentage_error'
        model.compile(loss={'output' : 'mae'},
                  optimizer=config['optimizer'],
                  metrics=[additional_metric_name, 'mse'])
    else:
        additional_metric_name = 'accuracy'
        model.compile(loss={'output' : 'categorical_crossentropy'},
                  optimizer=config['optimizer'],
                  metrics=[additional_metric_name])
    
    json.dump(json.loads(model.to_json()), 
              open(config['base_folder'] + 'classification/model_lstm_' + task + '.json','w'), indent=4, sort_keys=True)   

    
    early_stopping = EarlyStopping(monitor='val_loss' if task == 'los' else 'val_acc', patience=10)
    visualizer = LossHistoryVisualisation(config['base_folder'] + 'classification/epochs_' + task + '.png', additional_metric_name='val_' + additional_metric_name)
    model.fit({'codes_input':codes_train, 'demo_input':demo_train}, {'output':y_train},
              nb_epoch=50,
              validation_data=({'codes_input':codes_validation, 'demo_input':demo_validation}, {'output':y_validation}),
              batch_size=64,
              verbose=2,
              callbacks=[early_stopping, visualizer])
    
    print("Prediction using LSTM..")
    score = model.evaluate({'codes_input':codes_test, 'demo_input':demo_test}, {'output':y_test}, verbose=0)
    
    
    print('Test MAE:', score[0])
    print('Test MAPE:', score[1])  

    return [model, score[1]]
