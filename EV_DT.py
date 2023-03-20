# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:20:49 2023

@author: chaub
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_data_clean.csv")
data.head()
data.columns
X = data[['Percentage_pop_with_bachelors', 'Adults 19-25',
       'Adults 26-34', 'Adults 35-54', 'EnergyC_addunit', 'num_jobs_tot',
       'Av_temperature', 'Median_income', 'GDP', 'Elec_price', 'Tot_vehicle']]
y = data[['Label_py']]
print(y)
print(y.Label_py.unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_features = X.columns

y_feature = ['0','1','2']



#Using Gini
dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 4)
dt.fit(X_train, y_train)

preds = dt.predict(X_test)

print('Confusion matrix for Decision Trees with gini')
print(confusion_matrix(y_test, preds))

print('Classification report for Decision Tree with gini')
print(classification_report(y_test, preds))


from sklearn import tree
text_representation = tree.export_text(dt)
print(text_representation)

fig = plt.figure(figsize = (30,6))
decision_tree= tree.plot_tree(dt, feature_names = X_features, class_names = y_feature, filled = True)
print(decision_tree)


#Using Entropy

dt_2 = DecisionTreeClassifier(splitter = 'best',criterion = 'entropy', max_depth = 4)
dt_2.fit(X_train, y_train)

preds_2 = dt_2.predict(X_test)

print('Confusion matrix for Decision Trees with entropy')
print(confusion_matrix(y_test, preds_2))

print('Classification report for Decision Tree with entropy')
print(classification_report(y_test, preds_2))

text_representation_2 = tree.export_text(dt_2)
print(text_representation_2)

fig = plt.figure(figsize = (40,6))
decision_tree_2= tree.plot_tree(dt_2, feature_names = X_features, class_names = y_feature, filled = True)
print(decision_tree_2)


#Using entropy and random with max depth 6
dt_3 = DecisionTreeClassifier(splitter = 'best',criterion = 'entropy', max_depth = 6)
dt_3.fit(X_train, y_train)

preds_3 = dt_3.predict(X_test)

print('Confusion matrix for Decision Trees with entropy')
print(confusion_matrix(y_test, preds_3))

print('Classification report for Decision Tree with entropy')
print(classification_report(y_test, preds_3))

text_representation_3 = tree.export_text(dt_3)
print(text_representation_3)

fig = plt.figure(figsize = (30,10))
decision_tree_3= tree.plot_tree(dt_3, feature_names = X_features, class_names = y_feature, filled = True)
print(decision_tree_3)
plt.savefig("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/DT_6.png", dpi = 600)



#### Building decision tree by removing uneccesary features
X = data[['Percentage_pop_with_bachelors',
       'Adults 26-34', 
       'Av_temperature', 'Median_income', 'GDP', 'Elec_price']]
y = data[['Label_py']]
print(y)
print(y.Label_py.unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_features = X.columns

y_feature = ['0','1','2']

dt_4 = DecisionTreeClassifier(splitter = 'best',criterion = 'entropy', max_depth = 4)
dt_4.fit(X_train, y_train)

preds_4 = dt_4.predict(X_test)

print('Confusion matrix for Decision Trees with entropy')
print(confusion_matrix(y_test, preds_4))

print('Classification report for Decision Tree with entropy')
print(classification_report(y_test, preds_4))

text_representation_4 = tree.export_text(dt_4)
print(text_representation_4)

fig = plt.figure(figsize = (40, 7))
decision_tree_4= tree.plot_tree(dt_4, feature_names = X_features, class_names = y_feature, filled = True)
print(decision_tree_4)
plt.savefig("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/DT_reduced_4.png", dpi = 600)
