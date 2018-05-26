# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:20:47 2018

@author: lnc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:27:20 2018

@author: lnc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:19:11 2018

@author: lnc
"""

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url="C:/Hackathon/TD1.csv"
names = ['Gender', 'Insurer Id', 'No Of Days', 'Claims Type', 'Claims Charge','Age','class']
dataset = pandas.read_csv(url, names=names)
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
array = dataset.values
X = array[:,0:6]
print("X = {}".format(X))
Y = array[:,6]
print("Y = {}".format(Y))
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric

#print("Xtrain_{} \n\n Y_train {}".format(X_train,Y_train))
#print("X_validation{}".format(Y_train))
#print("X_validation{}".format(X_validation))
#print("X_train{}".format(len(X_train)))

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("predictions {}".format(predictions))
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))