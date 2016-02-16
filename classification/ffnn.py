from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

def train_and_evaluate_ffnn(config, X_train, X_test, y_train, y_test, output_dim):
    y_train = np_utils.to_categorical(y_train, output_dim)
    y_test = np_utils.to_categorical(y_test, output_dim)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=23)
    scaler = preprocessing.MaxAbsScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    scaler.transform(X_validation)
    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(50, input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, init='uniform'))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd)
    
    model.fit(X_train, y_train,
              nb_epoch=40,
              batch_size=16,
              show_accuracy=True,
              validation_data=(X_validation, y_validation),
              verbose=2)
    
    print("Prediction using FFNN..")
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  
    
    return [model, score[1]]

