# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:56:40 2018

@author: hp
"""

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import pandas
import configparser
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
#url="G:/Python/Hackathon-MyCopy/TD1.csv"
#url="E:/GitHub Projs/Hackathon/TD1.csv"
#names = ['Gender', 'Insurer Id', 'No Of Days', 'Claims Type', 'Claims Charge','Age','class']
#dataset = pandas.read_csv(url, names=names)
config = configparser.ConfigParser()
config.read("E:/GitHub Projs/Hackathon/properties.ini")
dataset = pandas.read_csv(config["FileInfo"]["filePath"])
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
#print(dataset.groupby('class').size())
print(dataset.groupby('Class').size())
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
array = dataset.values
X = array[0:20:,0:6]
#print("X = {}".format(X))
Y = array[0:20:,6]
xValidation = array[20:,0:6]
yvalidation = array[20:,6]
print("train Data {}".format(X))
print("TestData = {}".format(xValidation))
knn = KNeighborsClassifier()
knn.fit(X, Y)
predictions = knn.predict(xValidation)
print("predictions {}".format(predictions))
print("accuracy {}".format(accuracy_score(yvalidation, predictions)))