# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.metrics


datas = pd.read_csv(open("C:\\Users\\ashis\\Desktop\\datat.csv"))

datas['sex'] = datas['sex'].map({'M': 0, 'F': 1})
datas['address'] = datas['address'].map({'U': 0, 'R': 1})
datas['guardian'] = datas['guardian'].map({'mother': 0, 'father': 1})


predictors = datas.values[:, 0:11]
targets = datas.values[:,12]


pred_train, pred_test, targ_train, targ_test = train_test_split(predictors, targets, test_size=0.33)

clf =svm.SVC(kernel='rbf', C=1000, gamma=1000)
clf.fit(pred_train,targ_train)

pred = clf.predict(pred_test)

#accuracy
print("Accuracy is",accuracy_score(targ_test, pred, normalize = True))
#classification error
print("Classification error is",1- accuracy_score(targ_test, pred, normalize = True))
#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(targ_test, pred, labels=None, average =  'micro', sample_weight=None))
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(targ_test, pred,labels=None, average =  'micro', sample_weight=None))