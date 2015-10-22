# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
import sklearn.metrics as metrics

iris = datasets.load_iris()

svc = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(svc, iris.data, iris.target, cv=5, scoring='f1_macro')

print(scores)