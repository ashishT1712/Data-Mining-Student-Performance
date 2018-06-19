import pandas as pd
import sklearn.metrics
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

adult_df = pd.read_csv(r'''C:\Users\unish\Desktop\datat.csv''')
adult_df.columns = ['sex','age','Medu','Fedu','studytime','failures','famrel','freetime','health','absences','address','guardian','G3' ]

#print(adult_df.isnull().sum())

'''for value in ['sex' ]:
    print (value,":", sum(adult_df[value] == '?'))'''

adult_df_rev = adult_df

le = preprocessing.LabelEncoder()
sex_cat = le.fit_transform(adult_df.sex)
address_cat = le.fit_transform(adult_df.address)
guardian_cat   = le.fit_transform(adult_df.guardian)

adult_df_rev['sex_cat'] = sex_cat
adult_df_rev['address_cat'] = address_cat
adult_df_rev['guardian_cat'] = guardian_cat

dummy_fields = ['sex','address','guardian']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)
#print(adult_df_rev.head())

adult_df_rev = adult_df_rev.reindex_axis(['sex_cat','age','Medu','Fedu','studytime','failures','famrel','freetime','health','absences','address_cat','guardian_cat','G3'], axis=1)

#print(adult_df_rev)

'''num_features = ['sex_cat','age','Medu','Fedu','studytime','failures','famrel','freetime','health','absences','address_cat','guardian_cat','G3']

scaled_features = {}
for each in num_features:
    mean, std = adult_df_rev[each].mean(), adult_df_rev[each].std()
    scaled_features[each] = [mean, std]
    adult_df_rev.loc[:, each] = (adult_df_rev[each] - mean) / std '''

#print(adult_df_rev)
features = adult_df_rev.values[:, :11]
target = adult_df_rev.values[:, 11]
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=10)

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)
print(target_pred)
#accuracy
print("Accuracy is",accuracy_score(target_test, target_pred, normalize = True))
#classification error
print("Classification error is",1- accuracy_score(target_test, target_pred, normalize = True))
#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(target_test, target_pred, labels=None, average =  'micro', sample_weight=None))
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(target_test, target_pred,labels=None, average =  'micro', sample_weight=None))

