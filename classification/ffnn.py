#from sknn.platform import cpu64, threads4
from sknn.platform import gpu64
# pip install scikit-neuralnetwork
from sklearn import preprocessing
from sknn.mlp import Classifier, Layer
from sklearn.cross_validation import train_test_split

def train_and_evaluate_ffnn(config, X_train, X_test, y_train, y_test):
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=23)
    scaler = preprocessing.MaxAbsScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    scaler.transform(X_validation)
    
    nn = Classifier(
    layers=[
        Layer("Rectifier", units=20),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=40,
    n_stable=5,
    verbose=True,
    valid_set= (X_validation, y_validation))
    
    print("Training FFNN..")
    try:
        nn.fit(X_train, y_train)
        # you can interrupt training with CTRL-C without stopping the script.
    except KeyboardInterrupt:
        pass
    
    print("Prediction using FFNN..")
    score = nn.score(X_test, y_test)
    print("Accuracy for classification task on the test set: " + str(score))
    
    
    return [nn, score]


