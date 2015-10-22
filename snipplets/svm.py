# -*- coding: utf-8 -*-
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
import numpy as np
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)    

plot(svc)

