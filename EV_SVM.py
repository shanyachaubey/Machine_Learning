# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:31:46 2023

@author: chaub
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data_raw = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_data_clean.csv")

print(data_raw.head())
print(data_raw.columns)


charge_data_import = pd.read_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_charging_data.csv') 

charge_data_import.head()

charge_data_import.columns

charging_melt = pd.melt(charge_data_import, id_vars = 'State', value_vars=['2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013','2012', '2011', '2010', '2009', '2008'])
charging_melt.head()

charging_melt['Key'] = charging_melt['State']+charging_melt['variable']

charging_melt['value'] = charging_melt['value'].astype(str)
print(charging_melt.loc[2]['value'])

def remove_comma(string):
    string =  re.sub(',','', string)
    return string

test = 'fing,nd'
print(remove_comma(test))

charging_melt['value'] = charging_melt['value'].apply(remove_comma)
print(charging_melt.loc[2]['value'])

charging_melt['value'] = charging_melt['value'].astype(int)

charging_melt = charging_melt[['value', 'Key']]

data = pd.merge(data_raw,charging_melt, how = 'left', on = 'Key')
data.head()

data.rename({'value':'charging_ports'}, axis = 1, inplace = True)
data.columns

data = data[['State', 'Percentage_pop_with_bachelors', 'Year', 'Adults 19-25',
       'Adults 26-34', 'Adults 35-54', 'EnergyC_addunit', 'num_jobs_tot',
       'Av_temperature', 'Median_income', 'GDP', 'Elec_price', 'Tot_vehicle',
       'Number of registrations', 'Key','charging_ports']]

data['Number of registrations'].quantile(0.5)
data['Number of registrations'].quantile(0.75)
data['Number of registrations'].quantile(0.80)
data['Number of registrations'].quantile(0.90)

#Creating a new column with percentage of electric vehicles in each state in each year.

data['perc_of_tot'] = (data['Number of registrations']*100)/data['Tot_vehicle']

#Scaling GDP by converting it to GDP per working population
data['Gdp_per_working_pop'] = data['GDP']/(data['Adults 19-25']+data['Adults 26-34']+data['Adults 35-54'])

#Scaling Energy consumption by dividing it by working population
data['Energy_per_working_pop'] = data['EnergyC_addunit']/(data['Adults 19-25']+data['Adults 26-34']+data['Adults 35-54'])

#Scaling Num jobs total by dividing it by working population
data['num_jobs_per_working_pop'] = data['num_jobs_tot']/(data['Adults 19-25']+data['Adults 26-34']+data['Adults 35-54'])

#Scaling charging ports total by dividing it by working population
data['charging_ports_per_w_pop'] = data['num_jobs_tot']/(data['Adults 19-25']+data['Adults 26-34']+data['Adults 35-54'])

data.columns

data_to_use = data[['State', 'Percentage_pop_with_bachelors', 'Year','Av_temperature','Median_income', 'Elec_price', 'Key', 'perc_of_tot', 'Gdp_per_working_pop', 'Energy_per_working_pop', 'num_jobs_per_working_pop', 'charging_ports_per_w_pop']]
data_to_use.head()

data_to_use.info()

sns.lineplot(data = data_to_use, x = 'Year', y = 'perc_of_tot', hue = 'State',legend = None)
plt.title('Trend of Electric vehicle registrations in US states 2008-2020')
plt.xlabel('Year')
plt.ylabel('Percentage of total number of vehicles')
#Checking the distribution of perc_of_tot

sns.distplot(data_to_use['perc_of_tot'], kde = True)
print(data_to_use.perc_of_tot.mean())
print(data_to_use.perc_of_tot.median())
print(data_to_use.perc_of_tot.mode())

print(data_to_use.perc_of_tot.quantile(0.75))
print(data_to_use.perc_of_tot.quantile(0.90))
print(data_to_use.perc_of_tot.max())
quant_75 = data_to_use.perc_of_tot.quantile(0.75)
quant_90 = data_to_use.perc_of_tot.quantile(0.90)
conditions = [(data_to_use['perc_of_tot']>0) & (data_to_use['perc_of_tot']<=quant_75),
              (data_to_use['perc_of_tot']>quant_75) & (data_to_use['perc_of_tot'] <= quant_90),
              data_to_use['perc_of_tot']>quant_90]
values = [0,1,2]
data_to_use['label'] = np.select(conditions, values)

pd.DataFrame.to_csv(data_to_use, 'C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_new.csv',index = False )

#Fitting SVM on the numerical features

#For this part we will experiment by including the year in analysis for one SVM
#and excluding it for another
data_to_use.columns
data_to_use.info()

#Creating x and y to split data
scaler = StandardScaler()

X_w_year = scaler.fit_transform(data_to_use[['Percentage_pop_with_bachelors', 'Year', 'Av_temperature',
       'Median_income', 'Elec_price', 'perc_of_tot',
       'Gdp_per_working_pop', 'Energy_per_working_pop',
       'num_jobs_per_working_pop', 'charging_ports_per_w_pop']])
X_scaled = pd.DataFrame(X_w_year, columns = ['Percentage_pop_with_bachelors', 'Year', 'Av_temperature',
       'Median_income', 'Elec_price', 'perc_of_tot',
       'Gdp_per_working_pop', 'Energy_per_working_pop',
       'num_jobs_per_working_pop', 'charging_ports_per_w_pop'])
y = np.array(data_to_use['label'])
len(y)

#splitting data into test and train
X_train_w_y, X_test_w_y, y_train_w_y, y_test_w_y = train_test_split(X_w_year, y, test_size=0.3)

print(f'The shape of training data set is:,{X_train_w_y.shape}')
print(f'The shape of training data set is:,{X_test_w_y.shape}')
print(f' Datatype of y is: {type(y)}')

#########Linear SVC, C = 10

svm_1 = LinearSVC(C=10, max_iter=10000)
svm_1.fit(X_train_w_y, y_train_w_y)

pred_1 = svm_1.predict(X_test_w_y)
print(f'The confusion matrix for linear svm is\n {confusion_matrix(y_test_w_y, pred_1)}')
print(f'The classification report for linear svc is: {classification_report(y_test_w_y, pred_1)}')


# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_1
labels = [0, 1, 2]
cm_1 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_1, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for linear kernel with c = 10')
plt.show()


#########Linear SVC, C = 20
svm_1_20 = LinearSVC(C=20, max_iter=10000)
svm_1_20.fit(X_train_w_y, y_train_w_y)

pred_1_20 = svm_1_20.predict(X_test_w_y)
print(f'The confusion matrix for linear svm is\n {confusion_matrix(y_test_w_y, pred_1_20)}')
print(f'The classification report for linear svc is: {classification_report(y_test_w_y, pred_1_20)}')
# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_1_20
labels = [0, 1, 2]
cm_1_20 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_1_20, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix  for linear kernel with c = 20')
plt.show()

#########Linear SVC, C = 30
svm_1_30 = LinearSVC(C=30, max_iter=10000)
svm_1_30.fit(X_train_w_y, y_train_w_y)

pred_1_30 = svm_1_30.predict(X_test_w_y)
print(f'The confusion matrix for linear svm is\n {confusion_matrix(y_test_w_y, pred_1_30)}')
print(f'The classification report for linear svc is: {classification_report(y_test_w_y, pred_1_30)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_1_30
labels = [0, 1, 2]
cm_1_30 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_1_30, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for linear kernel with c = 30')
plt.show()



#########Linear SVM using SVC instead of LinearSVC

svm_l = SVC(C = 2, kernel = 'linear', max_iter=(1100000), random_state=(123))
svm_l.fit(X_train_w_y, y_train_w_y)

pred_l = svm_l.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 2 is\n {confusion_matrix(y_test_w_y, pred_l)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_l)}')

print(X_w_year)


#using PCA to find which features to use for scatter plot
pca = PCA()
pca.fit_transform(X_scaled)

# plot explained variance ratio
variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(variance_ratio)
plt.plot(cumulative_variance_ratio)
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance Ratio')

# annotate with feature names and variance ratios
for i, (name, ratio) in enumerate(zip(['Percentage_pop_with_bachelors', 'Year', 'Av_temperature',
       'Median_income', 'Elec_price', 'perc_of_tot',
       'Gdp_per_working_pop', 'Energy_per_working_pop',
       'num_jobs_per_working_pop', 'charging_ports_per_w_pop'], variance_ratio)):
    plt.annotate(name + '\n({:.2%})'.format(ratio), (i+1, cumulative_variance_ratio[i]))

plt.show()

#**********************************************************************#

#Visualizing the above svm using year and percentage pop with bachelors

#**********************************************************************#




viz = X_scaled.loc[:, ['Percentage_pop_with_bachelors', 'Year']]
viz.shape

viz['label'] = y

sample = viz.sample(n=30)
sample.head()
sample.head()
X=sample.iloc[:,0:2]
y=sample['label']
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')

model = SVC(kernel='linear', C=1E10, max_iter=(100000))
model.fit(X, y)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')

plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1])

plt.show()



#########Polynomial SVM
#reshaping y

y_train_w_y = y_train_w_y.reshape(-1,1)
y_train_w_y = y_train_w_y.ravel()

y_test_w_y = y_test_w_y.reshape(-1,1)
y_test_w_y = y_test_w_y.ravel()

svm_2 = SVC(C = 2, kernel = 'poly', degree = 2, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_2.fit(X_train_w_y, y_train_w_y)

pred_2 = svm_2.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 2 is\n {confusion_matrix(y_test_w_y, pred_2)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_2)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_2
labels = [0, 1, 2]
cm_2 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_2, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for polynomial kernel with d = 2 and c = 2')
plt.show()
#########Polynomial with C = 20
svm_2_20 = SVC(C = 20, kernel = 'poly', degree = 2, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_2_20.fit(X_train_w_y, y_train_w_y)

pred_2_20 = svm_2_20.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 2 is\n {confusion_matrix(y_test_w_y, pred_2_20)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_2_20)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_2_20
labels = [0, 1, 2]
cm_2_20 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_2_20, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for polynomial kernel with d = 2 and c = 20')
plt.show()

#########Polynomial with C = 50
svm_2_50 = SVC(C = 50, kernel = 'poly', degree = 2, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_2_50.fit(X_train_w_y, y_train_w_y)

pred_2_50 = svm_2_50.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 2 is\n {confusion_matrix(y_test_w_y, pred_2_50)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_2_50)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_2_50
labels = [0, 1, 2]
cm_2_50 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_2_50, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for polynomial kernel with d = 2 and c = 50')
plt.show()


#########Polynomial with C = 50, d = 3
svm_3_50 = SVC(C = 50, kernel = 'poly', degree = 3, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_3_50.fit(X_train_w_y, y_train_w_y)

pred_3_50 = svm_3_50.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 3 is\n {confusion_matrix(y_test_w_y, pred_3_50)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_3_50)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_3_50
labels = [0, 1, 2]
cm_3_50 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_3_50, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for polynomial kernel with d = 3 and c = 50')
plt.show()


########Radial kernel
svm_4 = SVC(C = 1, kernel = 'rbf', degree = 3, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_4.fit(X_train_w_y, y_train_w_y)

pred_4 = svm_4.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 3 is\n {confusion_matrix(y_test_w_y, pred_4)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_4)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_4
labels = [0, 1, 2]
cm_4 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_4, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for rbf kernel with d = 3 and c = 1')
plt.show()

#########Radial kernel with c = 20

svm_4_20_20 = SVC(C = 1, kernel = 'rbf', degree = 3, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_4_20_20.fit(X_train_w_y, y_train_w_y)

pred_4_20 = svm_4_20_20.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 3 is\n {confusion_matrix(y_test_w_y, pred_4_20)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_4_20)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_4_20
labels = [0, 1, 2]
cm_4_20 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_4_20, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for rbf kernel with d = 3 and c = 20')
plt.show()

#########Radial kernel with c = 200

svm_4_200 = SVC(C = 1, kernel = 'rbf', degree = 3, gamma = 'auto', max_iter=(1100000), random_state=(123))
svm_4_200.fit(X_train_w_y, y_train_w_y)

pred_4_200 = svm_4_200.predict(X_test_w_y)
print(f'The confusion matrix for SVM with polynomial kernel of degree 3 is\n {confusion_matrix(y_test_w_y, pred_4_200)}')
print(f'The classification report for polynomial svm is: {classification_report(y_test_w_y, pred_4_200)}')

# Generate a confusion matrix
y_true = y_test_w_y
y_pred = pred_4_200
labels = [0, 1, 2]
cm_4_20 = confusion_matrix(y_true, y_pred, labels=labels)

# Visualize the confusion matrix using seaborn heatmap
sns.heatmap(cm_4_20, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for rbf kernel with d = 3 and c = 200')
plt.show()



