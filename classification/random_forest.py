from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_random_forest(config, data, targets):
    model = RandomForestClassifier(n_estimators=100, verbose=1, oob_score=True)
    print("Training Random Forest..")
    model.fit(data, targets)
    print("Predicting using Random Forest..")
    print(model.score(data, targets))
    
    
    return model
    