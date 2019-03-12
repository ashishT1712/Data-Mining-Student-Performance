# Student Performance prediction
## Machine Learning - Supervised Learning for student performance prediction 

The aim of this project is to improve the current trends in the higher education systems and to find out which factors might help in creating successful students. It is really necessary to find successful students as it motivates higher education systems to know them well and one way to know this is by using valid management and processing of the student’s database.

## Data Description
1. Data source link:  *http://archive.ics.uci.edu/ml/datasets/student+performance* 
2. Data format: Integer
3. Size:  396 rows X 33 columns
4. Number of Instances: 396
5. Number of Attributes: 33

This data is of student’s achievement in secondary education of Portuguese school. The data attributes include student grades, demographic, social and school related features) and it was collected by using questionnaires and school reports. Dataset are provided regarding the performance in subject: Mathematics. The target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade, while G1 and G2 correspond to the 1st and 2nd period grades. 

During the data pre-processing set we found out that data present in our dataset was clean, as a result we did not had to perform the data cleaning methods.

In our dataset we had 33 attributes and as result we had to reduce some of the attributes which were not so important, to get better accuracy and low-cost tree. In organizations these kind of strategies is performed to reduce the data, so we also decided to do the same.

## Decision tree 
A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences

![Decision Tree](https://github.com/ashishT1712/Data-Mining-Student-Performance/blob/master/DecisionTree.png)


## Naive Bayesian 
Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
![Naive Bayesian](https://github.com/ashishT1712/Data-Mining-Student-Performance/blob/master/NaiveBayesian.png)

## SVM 
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane.
![SVM](https://github.com/ashishT1712/Data-Mining-Student-Performance/blob/master/Support%20Vector%20Machine.png)

## K nearest neighbor  
In pattern recognition, the k-nearest neighbour’s algorithm (k-NN) is a non-parametric method used for classification and regression. 
![KNN](https://github.com/ashishT1712/Data-Mining-Student-Performance/blob/master/K-Nearest%20Neighbors.png)

We have implemented our algorithms with the help of Python. We have made use of in-built python libraries and packages to implement our classification algorithms. We have made use of the following libraries and packages:
1) Numpy
2) Pandas
3) Scikit-learn
4) Matplotlib

<img src= 'https://github.com/ashishT1712/Data-Mining-Student-Performance/blob/master/Comparision.png' width = 500px height = 500px />


__Highest Accuracy achieved = 80%__
