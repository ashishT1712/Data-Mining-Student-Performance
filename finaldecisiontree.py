import time
import graphviz
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score

start_time = time.time()


df = pd.read_csv(r'''C:\Users\unish\Desktop\datat.csv''')
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1})
#df['internet'] = df['internet'].map({'no': 0, 'yes': 1})


predictors = df.values[:, 0:11]
targets = df.values[:,12]

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size= 0.25)


print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)

features = list(df.columns[:11])

classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')
'''classifier=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
#max_depth=6, min_samples_leaf=7
#bagging = BaggingClassifier(DecisionTreeClassifier())'''
classifier = classifier.fit(pred_train,tar_train)

predictions = classifier.predict(pred_test)

print(sklearn.metrics.confusion_matrix(tar_test, predictions))

#classification accuracy
print("accuracy of training dataset is{:.2f}".format(classifier.score(pred_train,tar_train)))
print("accuracy of test dataset is {:.2f}".format(classifier.score(pred_test,tar_test)))
#print(accuracy_score(tar_test, predictions, normalize = True))

#error rate
print("Error rate is",1- accuracy_score(tar_test, predictions, normalize = True))

#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None))
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None))

#precision
#print("precision is",sklearn.metrics.precision_score(tar_test, predictions, labels=None, pos_label=1, average =  'micro' ,sample_weight=None))

#Recall


#time to execute
#print("time elapsed: {:.2f}s".format(time.time() - start_time))

dot_data = tree.export_graphviz(classifier, out_file='rabbit1',
                         feature_names= features,
                         class_names= None,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
print (graph)





