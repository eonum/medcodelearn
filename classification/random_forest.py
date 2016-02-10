from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_random_forest(config, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=50, verbose=1, oob_score=True, n_jobs=config['num-cores'], random_state=23)
    print("Training Random Forest..")
    model.fit(X_train, y_train)
    print("Predicting using Random Forest..")
    score = model.score(X_test, y_test)
    print("Accuraccy for classification task on a test set: " + str(score))
    
    
    return [model, score]
    