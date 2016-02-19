from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from classification.LossHistoryVisualization import LossHistoryVisualisation
from keras.utils import np_utils

def train_and_evaluate_ffnn_word2vec(config, X_train, X1_test, y_train, y1_test, y2_train, y2_test, task, output_dim): 
    X1_train, X1_validation, y1_train, y1_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=23)
    scaler = preprocessing.MaxAbsScaler().fit(X_train)
    X1_train = scaler.transform(X1_train)
    X1_test = scaler.transform(X1_test)
    X1_validation = scaler.transform(X1_validation)
    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, input_dim=X1_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y1_train.shape[1], activation='sigmoid'))
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss="mean_squared_error", optimizer = "sgd")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    visualizer = LossHistoryVisualisation(config['base_folder'] + 'classification/epochs_' + task + '.png')
    model.fit(X1_train, y1_train,
              nb_epoch=35,
              batch_size=16,
              show_accuracy=True,
              validation_data=(X1_validation, y1_validation),
              verbose=2,
              callbacks=[early_stopping, visualizer])
    
    print("Prediction using FFNN..")
    score = model.evaluate(X1_test, y1_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  
    
    X2_test = model.predict(X1_test, batch_size=128, verbose=0)
    X2_train = y_train #model.predict(X_train, batch_size=128, verbose=0)
    y2_train = np_utils.to_categorical(y2_train, output_dim)
    y2_test = np_utils.to_categorical(y2_test, output_dim)
    X2_train, X2_validation, y2_train, y2_validation = train_test_split(X2_train, y2_train, test_size=0.15, random_state=23)

    print(str(X2_test.shape))
    print(str(X2_train.shape))
    print(str(X2_validation.shape))
    print(str(y2_test.shape))
    print(str(y2_train.shape))
    print(str(y2_validation.shape))

    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, input_dim=X2_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta')
    
    early_stopping = EarlyStopping(monitor='val_acc', patience=10)
    visualizer = LossHistoryVisualisation(config['base_folder'] + 'classification/epochs_word2vectransformer_' + task + '.png')
    model.fit(X2_train, y2_train,
              nb_epoch=35,
              batch_size=16,
              show_accuracy=True,
              validation_data=(X2_validation, y2_validation),
              verbose=2,
              callbacks=[early_stopping, visualizer])
    
    print("Prediction using FFNN Transform..")
    score = model.evaluate(X2_test, y2_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  
    
    return [model, score[1]]


