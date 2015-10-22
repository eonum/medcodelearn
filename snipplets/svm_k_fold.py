# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)


kf = cross_validation.KFold(150, n_folds=5, shuffle=True)

for train_index, test_index in kf:
    iris_X_train, iris_X_test = iris_X[train_index], iris_X[test_index]
    iris_y_train, iris_y_test = iris_y[train_index], iris_y[test_index]

    svc = svm.SVC(kernel='linear')
    svc.fit(iris_X_train, iris_y_train)    

    print(svc.predict(iris_X_test))
    print(iris_y_test) 

    
    input("Press Enter to continue...")

