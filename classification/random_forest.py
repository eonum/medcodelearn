from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

def train_and_evaluate_random_forest(config, data, targets):
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.33, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, verbose=1, oob_score=True, n_jobs=config['num-cores'])
    print("Training Random Forest..")
    model.fit(X_train, y_train)
    print("Predicting using Random Forest..")
    score = model.score(X_test, y_test)
    print("Accuraccy for code proposal task on a test set: " + str(score))
    
    
    return [model, score]
    