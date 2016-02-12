# pip install scikit-neuralnetwork
from sknn.mlp import Classifier, Layer

def train_and_evaluate_ffnn(config, X_train, X_test, y_train, y_test):
    nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=40,
    verbose=True,
    # TODO: Further divide into a validation set
    valid_set= (X_test, y_test))
    
    print("Training FFNN..")
    nn.fit(X_train, y_train)
    
    print("Prediction using FFNN..")
    score = nn.score(X_test, y_test)
    print("Accuracy for classification task on the test set: " + str(score))
    
    
    return [nn, score]


