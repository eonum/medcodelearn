# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
import sklearn.metrics as metrics

from reader.sparsehierarchical.drgreader import DRGCodeProposalReader 

r = DRGCodeProposalReader('data/2015/trainingData2015_20151001.csv')
training_set = r.read_from_file()

d = training_set.get_hierarchically_coded_feature_vectors_and_targets()

data = []
target = []

for elem in d:
    data.append(elem[0])
    target.append(elem[1])

data = np.array(data)
target = np.array(target)

# TODO: Normalization

svc = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(svc, data, target, cv=5, scoring='f1_macro')

print(scores)