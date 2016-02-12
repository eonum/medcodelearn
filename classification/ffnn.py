# pip install scikit-neuralnetwork
from sknn.mlp import Classifier, Layer

def train_and_evaluate_ffnn(config, X_train, X_test, y_train, y_test):
    nn = Classifier(
    layers=[
        Layer("Maxout", units=100, pieces=2),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=25)
    
    print("Training FFNN..")
    nn.fit(X_train, y_train)
    
    print("Prediction using FFNN..")
    score = nn.score(X_test, y_test)
    print("Accuracy for classification task on the test set: " + str(score))
    
    
    return [nn, score]


