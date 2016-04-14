from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from classification.LossHistoryVisualization import LossHistoryVisualisation
import json

def train_and_evaluate_ffnn(config, X_train, X_test, y_train, y_test, output_dim, task):
    y_train = np_utils.to_categorical(y_train, output_dim)
    y_test = np_utils.to_categorical(y_test, output_dim)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=23)
    scaler = preprocessing.MaxAbsScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_validation = scaler.transform(X_validation)
    
    model = Sequential()
    
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    json.dump(json.loads(model.to_json()), 
              open(config['base_folder'] + 'classification/model_' + task + '.json','w'), indent=4, sort_keys=True)   

    
    early_stopping = EarlyStopping(monitor='val_acc', patience=10)
    visualizer = LossHistoryVisualisation(config['base_folder'] + 'classification/epochs_' + task + '.png')
    model.fit(X_train, y_train,
              nb_epoch=100,
              batch_size=128,
              validation_data=(X_validation, y_validation),
              verbose=2,
              callbacks=[early_stopping, visualizer])
    
    print("Prediction using FFNN..")
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  
    
    return [model, scaler, score[1]]

